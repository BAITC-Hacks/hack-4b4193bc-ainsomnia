"""Лист заданий генератора A — раздел 5n CLAUDE.md.

    .venv/bin/python -m src.synth.tasks        # -> data/synth/tasks_a.jsonl, seeds.json

Каждая строка листа — условия одного синтетического обращения: класс, опорное
значение справочника, язык, регистр, роль, место, дата, длина, флаги. Тексты
пишутся строго по строкам (генератор A — Claude в сессии, без внешнего API),
поэтому разнообразие задаёт сетка условий, а не просьба «пиши разнообразно».

Из реальных данных берутся только агрегаты: значения справочника, встречающиеся
не реже 20 раз, география (районы и населённые пункты), сезонность тем и
распределение тем по регионам. Свободный текст не читается вовсе.
Метка — map_topic() опорного значения: та же функция, что размечает данные.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.topic_mapping import TOPICS, map_topic, norm

OUT = Path("data/synth")
SEED = 42
MIN_ROWS = 20                   # опорное значение — не реже 20 строк (5n)
N_RU, N_KK, N_MIX = 150, 50, 10  # на класс: 150 рус., 60 каз., из них 10 смешанных
N_INCIDENTS, INCIDENT_SIZE = 20, 5
HOLDOUT_SHARE, HOLDOUT_MIN_SEEDS = 0.20, 3
OPERATOR_SHARE = 0.75           # запись оператора по звонку (каналы, 5n)
EXPLICIT_SHARE = BORDER_SHARE = 0.10

# «Прочее» — только значения, называющие предмет. Формы обращения предмета не
# называют: из них класс «вне 14 тем» превратился бы в «мусор» (5n).
FORM_VALUES = {"жалобы", "жалоба", "иное", "предложения", "предложение", "заявление",
               "заявления", "запрос", "обращение", "прочее", "другое", "разное"}

# Ручной просмотр опорных значений (5n, 2026-09-26). Метка — map_topic(), и
# по значениям, которые она размечает неверно, текст противоречил бы метке.
# Такие значения не используются; разметка реальных данных не меняется —
# ошибки записаны в раздел 9 CLAUDE.md. Ключ словаря — начало значения.
SUBSTR = "ошибка разметки: ключ внутри другого слова"
CONTEXT = "ошибка разметки: ключ в чужом контексте"
MIXED = "смешанная категория: несколько тем сразу"
MISSED = "по смыслу — одна из 14 тем, правила её не поймали"
VAGUE = "не называет предмет обращения"
EXCLUDE = {
    "Определение делимости и неделимости": SUBSTR,          # «мост» в «делимости»
    "Пассажирский железнодорожный транспорт": SUBSTR,       # «дорожн»
    "Сбор и вывоз твердых бытовых отходов с Восточной": SUBSTR,  # «сточн»
    "Выявление бесхозяйных земельных участков": SUBSTR,     # «остановк»
    "Очистка территории от мусора с погрузкой": CONTEXT,    # «перевоз» -> транспорт
    "Лежачий полицейский": CONTEXT,                         # «полиц»: это дорога
    "Обеспечение безопасности пищевой продукции": CONTEXT,  # «безопасн»
    "Обеспечение безопасности водохозяйственных": CONTEXT,
    "Спортивная, детская площадка": CONTEXT,   # «спорт» -> соц.; «площадки» — благоустр.
    "Корректность и своевременность произведения начислений": MIXED,
    "Ликвидации чрезвычайных ситуаций (аварийные деревья": MIXED,
    "Осуществление материально-технического обеспечения": VAGUE,
    # «прочее»: по смыслу одна из 14 тем
    "колонка": MISSED, "Теплоизоляция труб": MISSED, "Содержание остановичных": MISSED,
    "Замена, ремонт, установка, перекрытие задвижек": MISSED,
    "Доставка воды при авариях": MISSED, "104 (газ)": MISSED,
    "Изменение условий пользования услугами связи": MISSED,
    "Отправка, получение (бандеролей": MISSED, "Вакцинация": MISSED,
    "Закуп лекарственных средств": MISSED, "Рассмотрение жалоб на невыплату": MISSED,
    "Контроль за соблюдением трудового": MISSED, "Гидравлические испытания": MISSED,
    "Ликвидация аварий": MISSED, "Регулирование цен на услуги субъектов": MISSED,
    "Выдача технических условий": MISSED, "Некачественное ремонт домов": MISSED,
    "Защита прав интересов детей": MISSED, "Информация об отмене занятий": MISSED,
    "Лица без определенного места жительства": MISSED,
    # «прочее»: не называет предмет
    "ГУ «Аппарат Акима": VAGUE, "Взаимодействие с": VAGUE, "Оснащение": VAGUE,
    "Проведение проверок": VAGUE, "Профилактический контроль": VAGUE,
    "Техническое обследование": VAGUE, "Рассмотрение служебных документов": VAGUE,
    "Рассмотрение обращений физических": VAGUE, "Изучение и анализ работы": VAGUE,
    "Качество предоставляемых услуг": VAGUE, "Жалоба на сотрудника": VAGUE,
    "Жалоба на предоставление услуги": VAGUE, "Некачественные работы подрядных": VAGUE,
    "Осуществление оперативного и эксплуатационно": VAGUE, "Ввод в эксплуатацию": VAGUE,
    "Заключение на объекты": VAGUE, "Предоставление информации по Постановлению": VAGUE,
    "ОБЩЕСТВЕННОЕ РАЗВИТИЕ": VAGUE, "Внутренняя политика": VAGUE,
}


def excluded(value):
    """Причина исключения опорного значения или None."""
    for start, why in EXCLUDE.items():
        if str(value).startswith(start):
            return why
    return None


# Письменные каналы — сглаженные доли по Караганде (source) и Алматы
# (submittal_channel): соцсети, WhatsApp, Telegram, портал, приложение.
CHANNELS = {"соцсети": 0.55, "WhatsApp": 0.15, "Telegram": 0.10, "портал": 0.10,
            "приложение": 0.10}
PERSONAS = ["пенсионерка", "пенсионер", "мать двоих детей", "многодетный отец",
            "председатель ОСИ", "старшая по подъезду", "владелец магазина",
            "студент", "водитель такси", "житель частного сектора", "учительница",
            "медсестра", "человек с инвалидностью", "молодая семья", "фермер",
            "работник предприятия"]
SITUATIONS = ["с утра", "второй день", "третий день подряд", "уже неделю",
              "больше месяца", "с прошлого года", "после вчерашнего ветра",
              "каждый вечер", "повторное обращение", "третье обращение без ответа",
              "срочно: в доме маленькие дети", "срочно: опасно для людей"]
LENGTHS = {"оператор": {"короткий": 0.40, "средний": 0.45, "длинный": 0.15},
           "житель": {"короткий": 0.20, "средний": 0.45, "длинный": 0.35}}
STREETS = ["Абая", "Сатпаева", "Жамбыла", "Толе би", "Байтурсынова", "Ауэзова",
           "Гоголя", "Пушкина", "Мира", "Строителей", "Бухар жырау", "Республики",
           "Шокана Уалиханова", "Момышулы", "Абылай хана", "Жибек жолы", "Школьная",
           "Садовая", "Центральная", "Молодёжная", "Степная", "Лесная",
           "Кенесары", "Назарбаева", "Гагарина", "Космонавтов", "Торайгырова",
           "Естая", "Академика Бектурова", "Майры"]
# Где районов в выгрузке нет — общеизвестные населённые пункты (не из данных).
PUBLIC_PLACES = {"Павлодарская область": ["Павлодар", "Экибастуз", "Аксу",
                                          "село Ленинский"],
                 "Алматинская область": ["Конаев", "Талгар", "Есик", "Каскелен",
                                         "Отеген батыр"],
                 "Костанайская область": ["Костанай", "Рудный", "Лисаковск",
                                          "Тобыл"]}
# Легко путаемые соседи — для пограничных текстов (метка по главной теме).
NEIGHBOURS = {
    "ЖКХ": ["водоснабжение и канализация", "жилищный фонд", "теплоснабжение"],
    "водоснабжение и канализация": ["ЖКХ", "жилищный фонд", "дороги"],
    "теплоснабжение": ["ЖКХ", "жилищный фонд"],
    "электроснабжение и освещение": ["благоустройство и озеленение", "дороги"],
    "дороги": ["благоустройство и озеленение", "восстановление после земляных работ",
               "транспорт"],
    "транспорт": ["дороги", "социальные вопросы"],
    "благоустройство и озеленение": ["дороги", "вывоз мусора и санитария",
                                     "электроснабжение и освещение"],
    "вывоз мусора и санитария": ["благоустройство и озеленение", "экология"],
    "жилищный фонд": ["ЖКХ", "водоснабжение и канализация"],
    "восстановление после земляных работ": ["дороги", "благоустройство и озеленение"],
    "связь и телекоммуникации": ["электроснабжение и освещение", "социальные вопросы"],
    "экология": ["вывоз мусора и санитария", "благоустройство и озеленение"],
    "социальные вопросы": ["транспорт", "безопасность"],
    "безопасность": ["электроснабжение и освещение", "социальные вопросы"],
    "прочее": ["социальные вопросы", "ЖКХ"],
}


def load():
    df = pd.read_parquet("data/unified.parquet",
                         columns=["created_at", "region", "district", "category",
                                  "topic", "appeal_class"])
    return df[df.appeal_class == "problem"]


def seeds_by_topic(p):
    """Опорные значения: не реже MIN_ROWS строк, без различий регистра."""
    g = (p.dropna(subset=["category"]).groupby(["topic", "category"]).size()
          .rename("n").reset_index())
    g["key"] = g.category.map(norm)
    out, dropped = {}, {}
    for topic, t in g.groupby("topic"):
        if topic == "прочее":
            t = t[~t.key.isin(FORM_VALUES)]
        # одна запись на значение без учёта регистра: написание — самое частое,
        # число строк — сумма по всем написаниям
        agg = (t.sort_values("n", ascending=False).groupby("key")
                .agg(value=("category", "first"), n=("n", "sum")))
        vals = []
        for v, n in agg.sort_values("n", ascending=False).itertuples(index=False):
            if n < MIN_ROWS:
                continue
            if excluded(v):
                dropped[v] = [topic, int(n), excluded(v)]
                continue
            vals.append((v, int(n)))
        for v, _ in vals:     # метка задаётся той же функцией, что размечает данные
            assert map_topic(v) == topic, (v, map_topic(v), topic)
        out[topic] = vals
    return out, dropped


def places(p):
    """Место — реальные районы и населённые пункты выгрузки или общеизвестные."""
    def pretty(v):
        v = str(v).strip()
        if v.isupper():
            v = v.capitalize()
            if v.endswith("ский"):
                v += " район"
        return v
    res = {}
    for region in sorted(p.region.unique()):
        d = p.loc[(p.region == region) & p.district.notna(), "district"]
        vals = [pretty(v) for v in d.value_counts().index if str(v).strip()]
        if len(vals) <= 1:
            vals = PUBLIC_PLACES.get(region, vals)
        res[region] = vals[:25]
    return res


def weights(series):
    w = np.sqrt(series.astype(float))       # смягчение: иначе 69% — Павлодар
    return (w / w.sum()).to_dict()


def build():
    rng = np.random.default_rng(SEED)
    p = load()
    seeds, dropped = seeds_by_topic(p)
    geo = places(p)
    months = {t: (g.created_at.dt.month.value_counts(normalize=True).sort_index()
                  .reindex(range(1, 13), fill_value=0) + 1e-3).pipe(lambda s: s / s.sum())
              for t, g in p.groupby("topic")}
    reg_w = {t: weights(g.region.value_counts()) for t, g in p.groupby("topic")}

    holdout = {}
    for t, vals in seeds.items():
        names = [v for v, _ in vals]
        if len(names) >= HOLDOUT_MIN_SEEDS:
            k = max(1, round(HOLDOUT_SHARE * len(names)))
            holdout[t] = sorted(rng.choice(names, size=k, replace=False).tolist())

    def pick(d):
        keys = list(d)
        return keys[rng.choice(len(keys), p=np.array(list(d.values())) / sum(d.values()))]

    def date_for(t):
        m = int(rng.choice(range(1, 13), p=months[t].values))
        y = int(rng.choice([2024, 2025], p=[0.4, 0.6]))
        return pd.Timestamp(y, m, int(rng.integers(1, 29)))

    rows = []
    for ci, t in enumerate(TOPICS):
        names = [v for v, _ in seeds[t]]
        held = set(holdout.get(t, []))
        open_seeds = [v for v in names if v not in held]
        langs = ["ru"] * N_RU + ["kk"] * N_KK + ["kk-ru"] * N_MIX
        rng.shuffle(langs)
        slots = []
        for inc in range(N_INCIDENTS):          # происшествия — только открытые значения
            seed = open_seeds[inc % len(open_seeds)]
            region = pick(reg_w[t])
            base = {"incident": f"I-{ci:02d}-{inc:02d}", "seed": seed, "region": region,
                    "place": str(rng.choice(geo[region])),
                    "street": str(rng.choice(STREETS)), "day0": date_for(t)}
            for j in range(INCIDENT_SIZE):
                slots.append({**base, "day": base["day0"] + pd.Timedelta(
                    days=int(j + rng.integers(0, 2)))})
        for _ in range(len(langs) - len(slots)):
            region = pick(reg_w[t])
            slots.append({"incident": None, "seed": str(rng.choice(names)),
                          "region": region, "place": str(rng.choice(geo[region])),
                          "street": str(rng.choice(STREETS)) if rng.random() < 0.5 else None,
                          "day": date_for(t)})
        for n, (s, lang) in enumerate(zip(slots, langs)):
            reg = "оператор" if rng.random() < OPERATOR_SHARE else "житель"
            r = {"id": f"A-{ci:02d}-{n:03d}", "label": t, "lang": lang, "register": reg,
                 "channel": "звонок 109" if reg == "оператор" else pick(CHANNELS),
                 "persona": str(rng.choice(PERSONAS)),
                 "situation": str(rng.choice(SITUATIONS)),
                 "length": pick(LENGTHS[reg]), "seed": s["seed"],
                 "seed_holdout": s["seed"] in held, "region": s["region"],
                 "place": s["place"], "street": s["street"],
                 "date": s["day"].strftime("%Y-%m-%d"), "incident": s["incident"],
                 "explicit": bool(rng.random() < EXPLICIT_SHARE), "border": None}
            if rng.random() < BORDER_SHARE:
                r["border"] = str(rng.choice(NEIGHBOURS[t]))
            rows.append(r)
    return rows, seeds, holdout, geo, dropped


def main():
    rows, seeds, holdout, geo, dropped = build()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tasks_a.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "seeds.json").write_text(json.dumps(
        {"min_rows": MIN_ROWS, "form_values_excluded": sorted(FORM_VALUES),
         "seeds": {t: [[v, n] for v, n in vals] for t, vals in seeds.items()},
         "excluded": dropped, "holdout": holdout, "places": geo},
        ensure_ascii=False, indent=1),
        encoding="utf-8")
    df = pd.DataFrame(rows)
    print(f"заданий: {len(df)}; классов: {df.label.nunique()}")
    print(df.groupby("label").agg(всего=("id", "size"),
                                  рус=("lang", lambda s: (s == "ru").sum()),
                                  каз=("lang", lambda s: (s == "kk").sum()),
                                  смеш=("lang", lambda s: (s == "kk-ru").sum()),
                                  в_происшествиях=("incident", lambda s: s.notna().sum()),
                                  отложено=("seed_holdout", "sum"),
                                  опорных=("seed", "nunique")).to_string())
    print("\nрегистр:", df.register.value_counts().to_dict())
    print("каналы:", df.channel.value_counts().to_dict())
    print("явное название:", int(df.explicit.sum()), "| пограничных:", int(df.border.notna().sum()))
    print("\nотложенные значения:", {t: len(v) for t, v in holdout.items()})
    unused = [k for k in EXCLUDE if not any(v.startswith(k) for v in dropped)]
    print(f"исключено опорных значений: {len(dropped)}; правил исключения без "
          f"срабатывания: {unused}")


def show(ci, start, n):
    """Компактный вывод заданий для написания: класс ci, с позиции start, n штук."""
    rows = [r for r in map(json.loads, (OUT / "tasks_a.jsonl").open(encoding="utf-8"))
            if r["id"].startswith(f"A-{ci:02d}-")][start:start + n]
    print(f"# {rows[0]['label']}")
    for r in rows:
        flags = ("явно " if r["explicit"] else "") + (f"погран:{r['border']} " if r["border"] else "")
        where = r["place"] + (f", ул. {r['street']}" if r["street"] else "")
        print(f"{r['id']} {r['lang']:5} {r['register'][:3]}/{r['channel']:10} {r['length'][:3]} "
              f"| {r['persona']} | {r['situation']} | {where} ({r['region'].split()[0]}) "
              f"| {r['date']} | «{r['seed'][:60]}» {r['incident'] or ''} {flags}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--show"]:
        show(*map(int, sys.argv[2:5]))
    else:
        main()
