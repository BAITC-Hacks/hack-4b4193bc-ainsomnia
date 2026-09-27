"""Проверки синтетического корпуса — раздел 5n CLAUDE.md.

    .venv/bin/python -m src.synth.checks            # полный корпус генератора A
    .venv/bin/python -m src.synth.checks --partial  # во время написания пачек

Что проверяется у каждого текста:
  * ПДн — шаблоны src/export.py (телефон, 12 цифр, адрес «дом … кв») плюс 10–11
    цифр подряд, почта, ссылки, «ИИН»/«ЖСН»; имена по словарю, отчества по
    окончаниям, слова с фамильными окончаниями вне белого списка (улицы и места
    из листа заданий, известные слова);
  * опорное значение и название темы дословно — только если в задании нет
    флага explicit. Однословные значения («Лифт», «Пожар») не проверяются:
    без этого слова текст о лифте не написать;
  * длина — в своём диапазоне с допуском;
  * пропущенные и лишние id, точные дубли.

Срабатывание ПДн — текст удаляется и пишется заново (5n), не правится.
Синтетические тексты можно печатать; совпадения ПДн всё равно маскируются.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.export import ADDR, LONG_DIGITS, PHONE
from src.topic_mapping import norm

RAW = Path("data/synth/raw_a")
TASKS = Path("data/synth/tasks_a.jsonl")
LEN_RANGE = {"короткий": (40, 110), "средний": (110, 260), "длинный": (260, 520)}
LEN_SLACK = 0.25                 # допуск: генератор не считает символы точно

DIGITS_10 = re.compile(r"(?<!\d)\d{10,11}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.\w+")
URL = re.compile(r"https?://|www\.", re.I)
IIN_WORD = re.compile(r"\b(ИИН|ЖСН|БИН)\b", re.I)
PATRONYMIC = re.compile(r"\b[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+(вич|вна|ична|ұлы|қызы|улы|кызы)\b")
SURNAME = re.compile(r"\b[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]{2,}(ов|ова|ев|ева|ин|ина|ский|ская|енко|баев|баева)\b")
# Частые имена — русские и казахские; сравнение по началу слова с учётом падежей.
NAMES = """Александр Алексей Анатолий Андрей Антон Аркадий Борис Вадим Валентин Валерий
Василий Виктор Виталий Владимир Владислав Вячеслав Геннадий Георгий Григорий Дмитрий
Евгений Егор Иван Игорь Илья Кирилл Константин Леонид Максим Михаил Никита Николай
Олег Павел Пётр Роман Руслан Сергей Станислав Степан Тимур Фёдор Юрий Ярослав
Александра Алёна Алла Анастасия Ангелина Анна Валентина Валерия Вера Виктория Галина
Дарья Евгения Екатерина Елена Елизавета Жанна Зинаида Зоя Инна Ирина Карина Кристина
Ксения Лариса Лидия Любовь Людмила Маргарита Марина Мария Надежда Наталья Нина Оксана
Ольга Полина Раиса Светлана Софья Тамара Татьяна Юлия
Абай Адиль Азамат Айбек Айдар Айдос Алибек Алмас Алмат Арман Армат Аскар Асхат Батыр
Бауыржан Бекзат Берик Болат Даулет Дамир Данияр Ерболат Ерлан Ермек Жандос Жанибек
Ильяс Кайрат Канат Куаныш Мадияр Марат Мухтар Нурлан Нурсултан Олжас Рахат Рустем Самат
Санжар Серик Талгат Темирлан Улан Ерасыл Ернар Жасулан Мурат Бахыт Гани
Айгерим Айгуль Айжан Айнур Айсулу Акмарал Алия Асель Асем Ботагоз Гульмира Гульнара
Гульнур Динара Дана Жанар Жанара Жибек Зарина Индира Камила Карлыгаш Лаура Мадина
Макпал Меруерт Назерке Салтанат Сауле Толганай Томирис Улжан Шолпан Эльмира Aйдана
Айдана Балжан Гаухар Дильназ Сабина Самал Райхан Нургуль Роза Бибигуль""".split()
NAME_STEMS = sorted({n[:-1] if len(n) > 4 and n[-1] in "аяйь" else n for n in NAMES},
                    key=len, reverse=True)
NAME_RE = re.compile(r"\b(" + "|".join(map(re.escape, NAME_STEMS)) + r")[а-яёәғқңөұүһі]{0,3}\b")
# Слова с фамильными окончаниями, которые не ПДн: улицы, места, обычные слова.
ALLOW = {"абая", "гоголя", "пушкина", "сатпаева", "ауэзова", "байтурсынова",
         "торайгырова", "гагарина", "назарбаева", "кенесары", "момышулы", "естая",
         "бектурова", "майры", "лисаковск", "аршалынский", "целиноградский",
         "атбасарский", "зерендинский", "буландынский", "шортандинский", "бурабайский",
         "аккольский", "ерейментауский", "глубоковский", "самарский", "шемонаихинский",
         "уланский", "куршимский", "алтайский", "зайсанский", "тарбагатайский",
         "сайрамский", "толебийский", "жетысайский", "сарыагашский", "тюлькубасский",
         "казыгуртский", "ордабасинский", "байдибекский", "ленинский", "шахтинск",
         "шахтинский", "бухар-жырауский", "осакаровка", "осакаровский", "нуринский",
         "каркаралинский", "абайский", "шетский", "частный", "сельский", "городской",
         "районный", "областной", "детский", "российский", "казахский", "английский",
         "мужской", "женский", "жилой", "морозов", "домов", "заводской",
         "алматы", "абай", "жибек", "самал", "гражданин", "гражданина", "магазин",
         "магазина", "машина", "бензин",
         # казахские написания улиц листа заданий (фамилия в им. падеже)
         "сәтбаев", "байтұрсынов", "уәлиханов", "момышұлы", "әуезов", "торайғыров",
         "бектұров", "гагарин", "назарбаев", "пушкин", "гоголь", "жамбыл",
         "сатпаев", "байтурсынов", "уалиханов", "ауэзов", "торайгыров", "бектуров",
         "конаев", "қонаев", "мустафин", "мұстафин", "мустафина",  # города, посёлки
         # частые слова на -ов/-ев в начале фразы
         "ответов", "вопросов", "домов", "дворов", "столбов", "подвалов", "проводов",
         "районов", "заборов", "автобусов", "маршрутов", "специалистов", "работников",
         "документов", "контейнеров", "деревьев", "пешеходов", "сотрудников", "звонков",
         "часов", "раз", "месяцев", "сроков", "результатов", "жильцов", "отчётов", "отчетов", "кустов", "грузовиков", "платежей", "киосков"}


def pii_hits(text, allow_extra=()):
    """Причины подозрения на ПДн, пустой список — чисто."""
    allowed = ALLOW | {norm(w) for w in allow_extra}
    hits = []
    for name, rx in (("телефон", PHONE), ("12 цифр", LONG_DIGITS), ("10–11 цифр", DIGITS_10),
                     ("адрес дом/кв", ADDR), ("почта", EMAIL), ("ссылка", URL),
                     ("ИИН/ЖСН", IIN_WORD)):
        m = rx.search(text)
        if m and not (name == "адрес дом/кв" and not re.search(r"\bдом\b.*\bкв\b", text, re.I)):
            hits.append(f"{name}: {m.group(0)[:2]}***")
    for m in PATRONYMIC.finditer(text):   # «ул. Момышулы» — улица, не отчество
        if m.group(0).lower() not in allowed and "көше" not in text[m.end():m.end() + 8].lower():
            hits.append(f"отчество: {m.group(0)[:2]}***")
            break
    for m in NAME_RE.finditer(text):      # «Алматы», «Абая», «Жибек жолы» — не имена
        if m.group(0).lower() not in allowed:
            hits.append(f"имя: {m.group(0)[:2]}***")
            break
    for m in SURNAME.finditer(text):
        # «Сәтбаев көшесі», «ул. Гагарина»: фамилия в названии улицы — не ПДн
        after, before = text[m.end():m.end() + 8].lower(), text[max(0, m.start() - 12):m.start()].lower()
        if m.group(0).lower() in allowed or "көше" in after or re.search(r"(ул\.|улиц\w*|проспект\w*)\s*$", before):
            continue
        hits.append(f"фамилия?: {m.group(0)[:2]}***")
        break
    return hits


def seed_copy(text, task, class_name):
    """Опорное значение или название темы дословно — нарушение без explicit."""
    if task["explicit"]:
        return None
    t = norm(text)
    for phrase in (task["seed"], class_name):
        p = norm(phrase)
        if len(p.split()) >= 2 and p in t:
            return phrase
    return None


def load_tasks():
    return {r["id"]: r for r in map(json.loads, TASKS.open(encoding="utf-8"))}


def load_raw():
    out, duplicates = {}, []
    for f in sorted(RAW.glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["id"] in out:
                    duplicates.append(r["id"])
                out[r["id"]] = r["text"].strip()
    return out, duplicates


def check(raw, tasks, allow_extra=()):
    """Словарь id -> список проблем. allow_extra — места/улицы в белый список."""
    problems, seen = {}, {}
    for i, text in raw.items():
        t = tasks.get(i)
        if t is None:
            problems.setdefault(i, []).append("id нет в листе заданий")
            continue
        p = [f"ПДн {h}" for h in pii_hits(text, allow_extra)]
        c = seed_copy(text, t, t["label"])
        if c:
            p.append(f"дословно «{c}»")
        lo, hi = LEN_RANGE[t["length"]]
        if not lo * (1 - LEN_SLACK) <= len(text) <= hi * (1 + LEN_SLACK):
            p.append(f"длина {len(text)} вне {t['length']} {lo}–{hi}")
        key = norm(text)
        if key in seen:
            p.append(f"дубль {seen[key]}")
        seen[key] = i
        if p:
            problems[i] = p
    return problems


def main():
    tasks = load_tasks()
    raw, duplicates = load_raw()
    extra = {w for r in tasks.values() for s in (r["place"], r["street"] or "")
             for w in re.split(r"[\s,.-]+", s) if w}
    probs = check(raw, tasks, extra)
    partial = sys.argv[1:] == ["--partial"]
    if sys.argv[1:] and not partial:
        raise SystemExit("использование: python -m src.synth.checks [--partial]")
    done = {i[:4] for i in raw}
    missing = [i for i in tasks if i not in raw and (not partial or i[:4] in done)]
    print(f"написано {len(raw)} из {len(tasks)}; пропущено: {len(missing)}; повторных id: {len(duplicates)}")
    if missing:
        print("  пропуски:", ", ".join(missing[:20]))
    if duplicates:
        print("  повторные id:", ", ".join(duplicates[:20]))
    kinds = {}
    for v in probs.values():
        for x in v:
            kinds[x.split(":")[0].split(" «")[0]] = kinds.get(x.split(":")[0].split(" «")[0], 0) + 1
    print("проблем по видам:", kinds or "нет")
    for i, v in list(probs.items())[:60]:
        print(f"  {i}: {'; '.join(v)}")
    return int(bool((missing and not partial) or duplicates or probs))


if __name__ == "__main__":
    sys.exit(main())
