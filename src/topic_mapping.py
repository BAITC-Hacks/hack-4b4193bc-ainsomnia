#!/usr/bin/env python3
"""Общий слой тем и классов обращения.

    .venv/bin/python -m src.topic_mapping

Добавляет в data/unified.parquet две колонки:
  appeal_class — problem / info / system
  topic        — одна из 13 общих тем (только для appeal_class == 'problem'
                 эта разметка имеет смысл как метрика покрытия)

Правила — подстроки в названии темы, приведённом к нижнему регистру.
ПОРЯДОК ЗНАЧИМ: побеждает первое совпадение, поэтому специфичные правила
стоят выше общих («дворовое освещение» -> электроснабжение, не благоустройство).

Черновик для ручного разбора, не финальный справочник.
"""
import csv, sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

TOPICS = ["ЖКХ", "водоснабжение и канализация", "теплоснабжение",
          "электроснабжение и освещение", "дороги", "транспорт",
          "благоустройство и озеленение", "вывоз мусора и санитария",
          "жилищный фонд", "восстановление после земляных работ",
          "социальные вопросы", "безопасность", "прочее"]

# --- классы обращения. Проверяются ДО тем. system раньше info. ---
SYSTEM = ["сброс звонка", "cброс звонка", "срыв звонка", "зачитывание ответа",
          "call-cent", "call-центр", "call center", "переадресац", "102 (", "103 (",
          "112 (", "118,169", "запрос статуса", "контактные данные call",
          "контактные данные call-центра", "звонок не по адресу", "тестирован",
          "мониторинг", "качество работы екц", "благодарн", "оператор"]
INFO = ["справка", "справочн", "информирование населения", "карантинн",
        "время и место голосован", "выборы", "выбор ", "референдум", "голосован",
        "брачно-семейн", "государственное устройство", "правосуди", "статистик",
        "документирование и регистрац", "зачитыван"]

# --- (тема, ключевые слова), порядок значим ---
RULES = [
    ("восстановление после земляных работ", [
        "восстановление после производственных работ", "после производственных работ",
        "здания и сооружения", "земляных работ", "раскопк"]),
    ("водоснабжение и канализация", [
        "водоснаб", "водопровод", "канализац", "водоотвед", "питьев", "колодец",
        "люк", "утечка воды", "отсутствие воды", "нет воды", "напор воды",
        "качества холодной воды", "качество холодной воды", "холодной воды",
        "подтоплен", "подтаплива", "затоплен", "арык", "септик", "водоканал",
        "су арнасы", "сточн", "ливнёвк", "ливневк", "порыв", "утечк"]),
    ("теплоснабжение", [
        "теплоснаб", "отоплен", "котельн", "горячая вода", "горячего водоснаб",
        "теплотранзит", "теплов", "теплосет", "батаре", "радиатор", "жылу",
        "нет тепла", "отопительн", "гвс"]),
    ("электроснабжение и освещение", [
        "электроснаб", "электроэнерг", "электричес", "освещен", "фонар", "лампа",
        "лампоч", "кабель", "провод", "линия электро", "линий электроперед",
        "подстанц", "трансформат", "напряжен", "жарық", "энергоснаб",
        "отсутствие света", "нет света", "замыкан", "искрен", "столб", "опор"]),
    ("дороги", [
        "дорог", "дорожн", "тротуар", "асфальт", "проезжая часть", "светофор",
        "семафор", "разметк", "мост", "перекрест", "яма", "обочин", "путепровод",
        "пешеходн", "трасс", "искусственн неровност", "шлагбаум"]),
    ("транспорт", [
        "транспорт", "автобус", "маршрут", "остановк", "перевоз", "такси",
        "водитель", "кондуктор", "троллейбус", "трамва", "вокзал", "пассажир"]),
    ("вывоз мусора и санитария", [
        "мусор", "отход", "тбо", "свалк", "санитар", "сан чистк", "сан очистк",
        "санчистк", "дезинф", "дератиз", "ветеринар", "ветсервис", "бродяч",
        "отлов", "животн", "запах", "нечистот", "выгреб", "контейнер",
        "насеком", "грызун", "эпидемиолог"]),
    ("жилищный фонд", [
        "мжд", "кровл", "крыш", "подъезд", "лифт", "кондоминиум", "квартир",
        "жилищн", "жилой дом", "фасад", "подвал", "общедомов", "кск",
        "многоквартир", "аварийност", "ветхост", "жиль", "домофон", "пандус"]),
    ("ЖКХ", [
        "жкх", "коммунальн", "газоснаб", "газов", "утечка газа", "газа", "газ ",
        "жизнеобеспечен", "тариф", "начислен", "счётчик", "счетчик",
        "прибор учёта", "прибор учета", "единый платежный", "топлив"]),
    ("социальные вопросы", [
        "здравоохран", "медицин", "больниц", "поликлиник", "врач", "скорая",
        "образован", "школ", "детский сад", "детсад", "социальн", "пособ",
        "пенси", "занятост", "адресная помощь", "инвалид", "многодетн",
        "культур", "спорт", "трудоустрой", "зарплат", "мед.персонал", "осмс",
        "гобмп", "аптек", "травм"]),
    ("безопасность", [
        "полиц", "безопасн", "правопоряд", "видеонаблюден", "чрезвычайн",
        "пожар", "наркот", "преступ", "правонарушен", "хулиган", "чс",
        "мчс", "спасател", "террор", "общественного порядка"]),
    ("благоустройство и озеленение", [
        "благоустрой", "озелен", "дерев", "насажден", "трава", "траву", "трав",
        "кустарник", "парк", "сквер", "аллея", "площадк", "газон", "скамей",
        "ограждени", "бордюр", "снег", "наледь", "гололед", "дворов", "субботник",
        "клумб", "фонтан", "малые архитектурн", "городская среда", "уборка улиц",
        "уборка территор", "кронирован", "скос", "мебел", "общественные места"]),
]

def norm(v):
    return "" if v is None else str(v).strip().lower().replace("ё", "е")

def classify_appeal(value):
    v = norm(value)
    if not v:
        return "problem"
    for k in SYSTEM:
        if norm(k) in v:
            return "system"
    for k in INFO:
        if norm(k) in v:
            return "info"
    return "problem"

def map_topic(value):
    v = norm(value)
    if not v:
        return "прочее"
    for topic, keys in RULES:
        for k in keys:
            if norm(k) in v:
                return topic
    return "прочее"

def karaganda_bad_themes():
    """22 темы, встречающиеся только в 638 строках со сдвигом полей."""
    kar = Path("drive-download-20260907T161509Z-1-001/"
               "Обращения граждан 109 - Карагандинская область.csv")
    if not kar.exists():
        return set()
    csv.field_size_limit(10**9)
    GOOD = {"Быстрый ответ", "Письменное обращение", ""}
    rows = list(csv.DictReader(open(kar, encoding="utf-8-sig", newline="")))
    broken = {r["sub_category"] for r in rows if (r["answer_type"] or "") not in GOOD}
    clean = {r["sub_category"] for r in rows if (r["answer_type"] or "") in GOOD}
    return broken - clean

def audit_rules(df):
    """Для каждого правила: сколько тем оно ловит. Плюс поиск тем в «прочем»,
    отличающихся от ключевого слова на окончание или одно слово."""
    themes = df.loc[df.appeal_class == "problem", "category"].dropna().unique()
    other = [t for t in themes if map_topic(t) == "прочее"]
    print("\n=== ПОКРЫТИЕ ПРАВИЛ: сколько тем ловит каждое ключевое слово ===")
    dead = []
    for topic, keys in RULES:
        hits = {k: sum(1 for t in themes if norm(k) in norm(t)) for k in keys}
        live = {k: v for k, v in hits.items() if v}
        dead += [(topic, k) for k, v in hits.items() if not v]
        print(f"\n{topic}  — правил {len(keys)}, срабатывает {len(live)}")
        for k, v in sorted(live.items(), key=lambda x: -x[1])[:8]:
            print(f"    {v:4d} тем  «{k}»")
    print(f"\nПравил, не поймавших ни одной темы: {len(dead)}")
    print("   " + ", ".join(f"{k}" for _, k in dead[:25]))

    print("\n=== БЛИЗКИЕ ПРОМАХИ: темы в «прочем», похожие на существующее правило ===")
    allkeys = [(t, k) for t, keys in RULES for k in keys]
    found = []
    for th in other:
        nt = norm(th)
        for topic, k in allkeys:
            nk = norm(k)
            if nk in nt:
                continue
            best = max((SequenceMatcher(None, nk, w).ratio()
                        for w in nt.split()), default=0)
            whole = SequenceMatcher(None, nk, nt).ratio()
            r = max(best, whole)
            if r >= 0.78:
                found.append((r, th, k, topic))
    found.sort(reverse=True)
    seen = set()
    if not found:
        print("  не найдено")
    for r, th, k, topic in found:
        if th in seen:
            continue
        seen.add(th)
        n = int((df.category == th).sum())
        print(f"  {r:.2f}  {n:7d} строк  «{th[:52]}»  ~  правило «{k}» ({topic})")

def main():
    src = Path("data/unified.parquet")
    if not src.exists():
        sys.exit("нет data/unified.parquet — сначала src/adapters/build_unified.py")
    df = pd.read_parquet(src)

    bad = karaganda_bad_themes()
    mask_bad = (df.region == "Карагандинская область") & df.category.isin(bad)
    print(f"Караганда: {len(bad)} тем только из строк со сдвигом полей — "
          f"{int(mask_bad.sum())} строк исключено из разметки\n")

    df["appeal_class"] = df.category.map(classify_appeal)
    df.loc[mask_bad, "appeal_class"] = "system"
    df["topic"] = df.category.map(map_topic)
    df.to_parquet(src, index=False)
    print(f"{src} обновлён: добавлены appeal_class и topic\n")

    print("=== КЛАССЫ ОБРАЩЕНИЯ И ПОКРЫТИЕ ТЕМ ВНУТРИ problem ===")
    print(f"{'регион':32s} {'всего':>8s} {'problem':>8s} {'info':>8s} {'system':>7s} "
          f"{'info %':>7s} {'прочее в problem':>17s}")
    rows = []
    for r, g in df.groupby("region", sort=False):
        pr = g[g.appeal_class == "problem"]
        info, sysm = int((g.appeal_class == "info").sum()), int((g.appeal_class == "system").sum())
        oth = int((pr.topic == "прочее").sum())
        share = oth / len(pr) if len(pr) else 0
        flag = "" if share < 0.15 else "  <-- >15%"
        print(f"{r:32s} {len(g):8d} {len(pr):8d} {info:8d} {sysm:7d} "
              f"{100*info/len(g):6.1f}% {100*share:16.1f}%{flag}")
        rows.append((r, len(g), len(pr), info, sysm, share))
    tot_pr = df[df.appeal_class == "problem"]
    print(f"\nВСЕГО {len(df)}: problem {len(tot_pr)} "
          f"({100*len(tot_pr)/len(df):.1f}%), "
          f"info {int((df.appeal_class=='info').sum())} "
          f"({100*(df.appeal_class=='info').mean():.1f}%), "
          f"system {int((df.appeal_class=='system').sum())} "
          f"({100*(df.appeal_class=='system').mean():.1f}%)")
    print(f"Прочее внутри problem по всей таблице: "
          f"{100*(tot_pr.topic=='прочее').mean():.1f}%")

    print("\n=== РАСПРЕДЕЛЕНИЕ ПО 13 ТЕМАМ (только problem) ===")
    vc = tot_pr.topic.value_counts()
    for t in TOPICS:
        n = int(vc.get(t, 0))
        print(f"  {t:36s} {n:8d}  {100*n/len(tot_pr):5.2f}%")

    audit_rules(df)

    print("\n=== ТОП-15 ТЕМ В «ПРОЧЕМ» ВНУТРИ problem, ПО РЕГИОНАМ ===")
    for r, g in df.groupby("region", sort=False):
        oth = g[(g.appeal_class == "problem") & (g.topic == "прочее")]
        if not len(oth):
            continue
        pr = int((g.appeal_class == "problem").sum())
        print(f"\n--- {r}: {len(oth)} строк ({100*len(oth)/pr:.1f}% от problem)")
        for v, k in Counter(oth.category.fillna("(пусто)")).most_common(15):
            print(f"  {k:7d}  {str(v)[:66]}")

if __name__ == "__main__":
    main()
