#!/usr/bin/env python3
"""Вопросы к данным 109 на естественном языке.

    .venv/bin/python -m src.nlq "сколько обращений было в прошлом месяце"
    .venv/bin/python -m tests.test_nlq          # прогон по 20 вопросам

Это запросы к СТРУКТУРИРОВАННОЙ таблице `data/unified.parquet`, а не поиск по
тексту: фильтры и агрегаты по дате, региону, теме, классу обращения, статусу.
Текста обращения в данных нет (раздел 5b CLAUDE.md), искать не в чем.

Разбор вопроса — детерминированный: словари значений берутся ИЗ ДАННЫХ, намерение
определяется по ключевым словам. Никакой языковой модели внутри нет намеренно:
вопрос, который не разобран, должен получить отказ с причиной, а не правдоподобное
число. Выдумывать значение, которого нет во входных данных, запрещено.

Каждый ответ состоит из трёх частей:
  ЧИСЛО   — результат
  ЗАПРОС  — как он получен, в читаемом виде
  ГРАФИК  — plotly-фигура (функции `fig_*` не зависят от streamlit)

Плюс ОГОВОРКИ, которые считаются из данных, а не прописаны под вопрос: заполненность
использованных полей, покрытие периода регионами, пропуски в выгрузке.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA = Path("data/unified.parquet")
FILL_WARN = 0.95        # поле заполнено ниже этой доли — оговорка обязательна
COVER_WARN = 0.50       # регион покрывает меньше этой доли периода — он не в счёте
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
             "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
DOW_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
HEAT_FROM, HEAT_TO = 10, 4     # отопительный сезон: октябрь → апрель

# Синонимы бытовой речи → тема канона. Это словарь предметной области, а не
# догадка о данных: каждое значение справа обязано существовать в колонке topic,
# что проверяется при загрузке (см. Vocab.check).
TOPIC_WORDS = {
    "вод": "водоснабжение и канализация", "канализац": "водоснабжение и канализация",
    "теплоснабж": "теплоснабжение", "отоплен": "теплоснабжение", "тепл": "теплоснабжение",
    "электроснабж": "электроснабжение и освещение", "свет": "электроснабжение и освещение",
    "освещен": "электроснабжение и освещение", "электр": "электроснабжение и освещение",
    "дорог": "дороги", "ям": "дороги",
    "мусор": "вывоз мусора и санитария", "санитар": "вывоз мусора и санитария",
    "благоустройств": "благоустройство и озеленение", "озеленен": "благоустройство и озеленение",
    "жкх": "ЖКХ", "жилищн": "жилищный фонд",
    "связ": "связь и телекоммуникации", "интернет": "связь и телекоммуникации",
    "эколог": "экология", "транспорт": "транспорт", "автобус": "транспорт",
    "безопасност": "безопасность", "социальн": "социальные вопросы",
    "земляны": "восстановление после земляных работ",
}
CLASS_WORDS = {
    "справочн": "info", "справк": "info", "информацион": "info",
    "проблем": "problem", "городск": "problem",
    "служебн": "system",
}
CLASS_RU = {"problem": "городские проблемы", "info": "справочные", "system": "служебные"}
# Разрезы по темам считаются по классу problem: info и system — справочный трафик и
# артефакты колл-центра, и без этого фильтра «прочее» выходит на первое место с 34%
# вместо 1.7% внутри problem. Раздел 5d CLAUDE.md. Фильтр всегда виден в запросе.
TOPIC_CUTS = ("top", "growth", "trend", "stopped", "share")
# «Прочее» — не тема, а остаток маппинга. В рейтинге тем оно отвечает не на вопрос
# руководителя, а на вопрос о качестве разметки, поэтому из рейтингов исключается —
# но не молча: его величина уходит в оговорку.
RESIDUAL_TOPIC = "прочее"
STOPPED_MONTHS = 6       # окно «перестало приходить»; период должен быть вдвое длиннее
GROWTH_MIN_BASE = 100    # меньше обращений в прошлом периоде — процент прироста не значит ничего
# Статусы в регионах написаны по-разному — это факт данных, не наша нормализация.
CLOSED_MARKS = ("closed", "закрыт")


def num(n):
    """Число с неразрывным разделением тысяч. Отдельной функцией, потому что
    .replace(",", " ") по всей строке ответа съедает запятые перечисления."""
    return f"{int(n):,}".replace(",", "\u00a0")


def norm(t):
    """Нижний регистр, ё→е, схлопнутые пробелы. Пунктуация заменяется пробелом,
    чтобы «теплоснабжению?» и «теплоснабжению» разбирались одинаково."""
    t = unicodedata.normalize("NFKC", str(t)).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


# ---------------------------------------------------------------- словари из данных
@dataclass
class Vocab:
    regions: list
    topics: list
    classes: list
    ref: pd.Timestamp                    # опорная дата = последняя дата в данных
    fill: dict                           # поле -> доля заполненности
    windows: dict                        # регион -> (первая дата, последняя)
    gaps: dict                           # регион -> (дней, начало, конец) пропуска

    @classmethod
    def build(cls, df):
        fill = {c: float(df[c].notna().mean()) for c in df.columns}
        windows, gaps = {}, {}
        for r, g in df.groupby("region"):
            lo, hi = g.created_at.min(), g.created_at.max()
            windows[r] = (lo, hi)
            d = g.set_index("created_at").resample("D").size()
            n, a, b = longest_zero_run(d)
            if n >= 7:
                gaps[r] = (n, a, b)
        v = cls(sorted(df.region.dropna().unique()),
                sorted(df.topic.dropna().unique()),
                sorted(df.appeal_class.dropna().unique()),
                df.created_at.max(), fill, windows, gaps)
        v.check()
        return v

    def check(self):
        """Словарь синонимов не должен ссылаться на тему, которой нет в данных."""
        missing = sorted({t for t in TOPIC_WORDS.values() if t not in self.topics})
        if missing:
            raise ValueError(f"в TOPIC_WORDS темы, которых нет в данных: {missing}")


def longest_zero_run(s):
    """Самая длинная череда точек с нулём: (длина, начало, конец)."""
    z = (s.to_numpy() == 0)
    best, i = (0, None, None), 0
    while i < len(z):
        if z[i]:
            j = i
            while j + 1 < len(z) and z[j + 1]:
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, s.index[i], s.index[j])
            i = j + 1
        else:
            i += 1
    return best


def load(path=DATA):
    from src.cli import BUILD_HINT, require
    require(path, "сводной таблицы обращений", BUILD_HINT)
    df = pd.read_parquet(path, columns=["created_at", "region", "district", "executor",
                                        "status", "sla_breach", "appeal_class", "topic"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    # Исполнитель бывает человеком — «ИП Фамилия» (раздел 1): в ответах только маской.
    from src.checks.person_names import safe
    ex = df["executor"].dropna().unique()
    df["executor"] = df["executor"].map({v: safe(v) for v in ex})
    return df, Vocab.build(df)


# ---------------------------------------------------------------- периоды
@dataclass
class Period:
    lo: pd.Timestamp
    hi: pd.Timestamp
    label: str

    def mask(self, df):
        return (df.created_at >= self.lo) & (df.created_at < self.hi)


def month_period(ts):
    lo = ts.to_period("M").to_timestamp()
    return Period(lo, lo + pd.offsets.MonthBegin(1), f"{MONTHS_RU[lo.month - 1]} {lo.year}")


def last_full_month(ref):
    return month_period(ref.to_period("M").to_timestamp() - pd.Timedelta(days=1))


def shift_year(p):
    lo = p.lo - pd.DateOffset(years=1)
    hi = p.hi - pd.DateOffset(years=1)
    lab = p.label
    if lo.day == 1 and (hi - lo).days <= 31:
        lab = f"{MONTHS_RU[lo.month - 1]} {lo.year}"
    else:
        lab = f"{lo:%Y-%m-%d} → {hi - pd.Timedelta(days=1):%Y-%m-%d}"
    return Period(lo, hi, lab)


def last_months(ref, n):
    hi = ref.to_period("M").to_timestamp()
    lo = hi - pd.DateOffset(months=n)
    return Period(lo, hi, f"последние {n} мес.: {lo:%Y-%m} → {hi - pd.Timedelta(days=1):%Y-%m}")


def quarter_period(ts):
    lo = ts.to_period("Q").to_timestamp()
    return Period(lo, lo + pd.offsets.QuarterBegin(1, startingMonth=1),
                  f"{lo.year} Q{lo.quarter}")


def last_quarter(ref):
    return quarter_period(ref.to_period("Q").to_timestamp() - pd.Timedelta(days=1))


def heating_season(ref):
    """Последний ЗАВЕРШЁННЫЙ отопительный сезон: октябрь → апрель."""
    y = ref.year if ref.month > HEAT_TO else ref.year - 1
    lo = pd.Timestamp(year=y - 1, month=HEAT_FROM, day=1)
    hi = pd.Timestamp(year=y, month=HEAT_TO + 1, day=1)
    if hi > ref:
        lo, hi = lo - pd.DateOffset(years=1), hi - pd.DateOffset(years=1)
    return Period(lo, hi, f"отопительный сезон {lo:%Y-%m} → {hi - pd.Timedelta(days=1):%Y-%m}")


def whole(df):
    return Period(df.created_at.min(), df.created_at.max() + pd.Timedelta(days=1), "вся история")


# ---------------------------------------------------------------- разбор вопроса
@dataclass
class Query:
    metric: str
    period: Period | None = None
    period_prev: Period | None = None
    filters: dict = field(default_factory=dict)
    group_by: str | None = None
    top: int | None = None
    focus: str | None = None        # названная в вопросе величина внутри group_by
    unit: str = "обращений"

    def where(self):
        parts = []
        for k, v in self.filters.items():
            parts.append(f"{k} = '{v}'")
        if self.period:
            parts.append(f"created_at >= '{self.period.lo:%Y-%m-%d}' "
                         f"AND created_at < '{self.period.hi:%Y-%m-%d}'")
        return parts

    def text(self):
        """Запрос в читаемом виде — печатается рядом с ответом."""
        if self.metric == "compare" and self.period_prev:
            base = [f"{k} = '{v}'" for k, v in self.filters.items()]
            def one(p):
                w = base + [f"created_at >= '{p.lo:%Y-%m-%d}' "
                            f"AND created_at < '{p.hi:%Y-%m-%d}'"]
                return ("SELECT COUNT(*) FROM обращения\n WHERE " + "\n   AND ".join(w))
            return (f"-- {self.period.label}\n{one(self.period)}\n\n"
                    f"-- {self.period_prev.label}\n{one(self.period_prev)}")
        agg = {"count": "COUNT(*)", "share": "COUNT(*), доля от целого",
               "avg_day": "COUNT(*) / число дней"}.get(self.metric, "COUNT(*)")
        s = f"SELECT {self.group_by + ', ' if self.group_by else ''}{agg}\n  FROM обращения"
        w = self.where()
        if w:
            s += "\n WHERE " + "\n   AND ".join(w)
        if self.group_by:
            s += f"\n GROUP BY {self.group_by}\n ORDER BY COUNT(*) DESC"
        if self.top:
            s += f"\n LIMIT {self.top}"
        return s


NUM_WORDS = {"один": 1, "два": 2, "три": 3, "трех": 3, "четыре": 4, "пять": 5,
             "шесть": 6, "семь": 7, "десять": 10}


def find_count(q, default):
    m = re.search(r"\b(\d+)\b", q)
    if m:
        return int(m.group(1))
    for w, n in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", q):
            return n
    return default


def find_topic(q):
    for key, topic in sorted(TOPIC_WORDS.items(), key=lambda kv: -len(kv[0])):
        if key in q:
            return topic
    return None


def find_class(q):
    for key, cl in CLASS_WORDS.items():
        if key in q:
            return cl
    return None


def find_region(q, vocab):
    for r in vocab.regions:
        stem = norm(r).split()[0][:8]          # «караганд», «павлодарск»
        if stem and stem in q:
            return r
    return None


def find_period(q, vocab, df):
    ref = vocab.ref
    if "всю историю" in q or "за все время" in q or "за всю" in q:
        return whole(df), None
    if "отопительн" in q:
        return heating_season(ref), None
    if "квартал" in q:
        cur = last_quarter(ref)
        prev = quarter_period(cur.lo - pd.Timedelta(days=1))
        return cur, prev
    m = re.search(r"последни\w* (\d+) месяц", q)
    if m:
        return last_months(ref, int(m.group(1))), None
    if "за последний год" in q or "за год" in q or "последний год" in q:
        return last_months(ref, 12), None
    if "месяц" in q:
        cur = last_full_month(ref)
        return cur, shift_year(cur)
    return None, None


def parse(question, vocab, df):
    """Вопрос → Query. Возвращает (Query, None) или (None, причина отказа).

    Поверх разбора действует одно правило: разрез по темам без явно названного
    класса считается по `problem`. Иначе «прочее» выходит на первое место с 34%
    (это справочный трафик и служебные артефакты), вместо 1.7% внутри problem.
    Фильтр добавляется в сам Query, поэтому виден в показанном запросе."""
    query, why = _parse(question, vocab, df)
    if query is None:
        return None, why
    touches_topic = query.group_by == "topic" or "topic" in query.filters
    if (query.metric in TOPIC_CUTS and touches_topic
            and "appeal_class" not in query.filters and not find_class(norm(question))):
        query.filters["appeal_class"] = "problem"
    return query, None


def _parse(question, vocab, df):
    q = norm(question)
    if not q:
        return None, "пустой вопрос"

    # Вопросы, на которые канон ответить не может. Отказ с причиной — правильный
    # ответ, а не провал: лучше молчание, чем правдоподобное число.
    for pat, why in (
        (r"сколько.*(длится|длилось)|врем\w* обработк|скольк\w* дней.*(обработ|закрыт)|"
         r"средн\w* срок",
         "в каноне нет даты закрытия обращения: у Караганды есть только updated_date — "
         "дата последнего изменения записи, и это открытый вопрос (раздел 3 CLAUDE.md). "
         "Посчитать длительность обработки не из чего"),
        (r"своими словами|текст обращени|о чем пишут|что пишут|жалуются.*словами",
         "текста обращения заявителя в выгрузке нет: 55 строк на 1 063 216, и каждая "
         "содержит ПДн (раздел 5b CLAUDE.md). Это запрос к владельцу данных, не к таблице"),
        (r"одного и того же заявител|повторн\w* обращени|один заявител|от одного человек",
         "идентификатор заявителя — персональные данные, в канон не переносится "
         "(раздел 1 CLAUDE.md). Связать обращения одного человека нечем"),
    ):
        if re.search(pat, q):
            return None, why

    period, prev = find_period(q, vocab, df)
    asked_class = find_class(q)
    filters = {}
    if (t := find_topic(q)):
        filters["topic"] = t
    if (r := find_region(q, vocab)):
        filters["region"] = r
    if asked_class and "доля" not in q:
        filters["appeal_class"] = asked_class

    # --- намерения, от самого узкого к самому общему
    if "прекратил" in q or "перестал" in q:
        p = period or last_months(vocab.ref, 2 * STOPPED_MONTHS * 2)
        # «Перестало приходить» = было до и нет после. Если период короче двух окон,
        # половина «до» пуста, и ответ «таких тем нет» получается всегда, при любых
        # данных. Это правдоподобное число вместо ответа — отказываем.
        if (p.hi - p.lo).days < 2 * STOPPED_MONTHS * 30:
            return None, (f"период короче {2 * STOPPED_MONTHS} месяцев, а вопрос "
                          f"«перестали приходить» сравнивает {STOPPED_MONTHS} месяцев "
                          f"до и {STOPPED_MONTHS} после. На таком периоде ответ «таких "
                          f"тем нет» получился бы при любых данных — спросите за год "
                          f"или больше")
        return Query("stopped", p, None, filters, group_by="topic"), None
    if "выходн" in q or "суббот" in q or "воскресен" in q:
        return Query("weekend", period or whole(df), None, filters), None
    if "дня недели" in q or "дни недели" in q or "день недели" in q:
        return Query("dow", period or whole(df), None, filters, group_by="день недели"), None
    # Пиковый день — только если спрошено про день или момент. Без этой оговорки
    # «больше всего обращений» перехватывает вопросы про темы и регионы.
    if (("когда" in q or "день" in q or "дата" in q or "пик" in q)
            and ("загружен" in q or "пик" in q or "больше всего" in q or "напряжен" in q)):
        return Query("peak_day", period or whole(df), None, filters), None
    if "в среднем в день" in q or "среднем за день" in q:
        return Query("avg_day", period or whole(df), None, filters, group_by="год"), None
    if ("росл" in q or "растут" in q or "вырос" in q or "прирост" in q
            or "прибав" in q or "увеличил" in q or "подрос" in q):
        if not prev:
            period, prev = last_months(vocab.ref, 12), shift_year(last_months(vocab.ref, 12))
        return Query("growth", period, prev, filters, group_by="topic", top=5), None
    if "как менялось" in q or "по месяцам" in q or "динамик" in q:
        return Query("trend", period or last_months(vocab.ref, 12), None, filters,
                     group_by="месяц"), None
    if "доля" in q or "процент" in q or "какая часть" in q or "много ли" in q:
        by = ("appeal_class" if (find_class(q) or "справочн" in q or "проблем" in q
                                 or "справк" in q)
              else "status" if "закрыт" in q
              else "sla_breach" if ("просроч" in q or "срок" in q or "нарушен" in q)
              else "topic")
        # Если тема названа прямо, вопрос про ЕЁ долю. Фильтр по теме при этом
        # снимается (иначе доля вышла бы 100%), но сама тема запоминается в focus:
        # без этого headline-числом становилась доля крупнейшей темы, то есть
        # ответ на другой вопрос.
        return Query("share", period or whole(df), None,
                     {k: v for k, v in filters.items() if k != by}, group_by=by,
                     top=find_count(q, None) if by == "topic" else None,
                     focus=filters.get(by)), None
    if "просроч" in q:
        return Query("share", period or whole(df), None, filters, group_by="sla_breach"), None
    if "закрыт" in q:
        return Query("share", period or whole(df), None, filters, group_by="status"), None
    if "район" in q or ("где" in q and "жалу" in q):
        return Query("top", period or whole(df), None, filters, group_by="district", top=5), None
    if "исполнител" in q or "кто получает" in q:
        return Query("top", period or whole(df), None, filters, group_by="executor", top=5), None
    if ("какие тем" in q or "каких тем" in q or "на что жалу" in q
            or ("тем" in q and ("больше всего" in q or "чаще всего" in q))
            or ("жалу" in q and "чаще всего" in q)):
        return Query("top", period or last_months(vocab.ref, 12), None, filters,
                     group_by="topic", top=find_count(q, 5)), None
    if "регион" in q or "област" in q:
        return Query("top", period or last_full_month(vocab.ref), None, filters,
                     group_by="region", top=7), None
    if "сколько" in q or "больше или меньше" in q:
        if prev and ("больше или меньше" in q or "год назад" in q or "по сравнению" in q):
            return Query("compare", period, prev, filters), None
        return Query("count", period or whole(df), None, filters), None
    return None, ("вопрос не разобран: не нашлось ни одной величины, которую можно "
                  "посчитать по таблице. Поддерживаются счёт, доля, топ, динамика, "
                  "прирост, день недели, пиковый день, среднее в день")


# ---------------------------------------------------------------- исполнение
def apply_filters(df, filters):
    m = pd.Series(True, index=df.index)
    for k, v in filters.items():
        m &= (df[k] == v)
    return df[m]


def drop_residual(g, query, res):
    """Убрать «прочее» из рейтинга тем, назвав его величину в оговорке."""
    if query.group_by != "topic" or RESIDUAL_TOPIC not in g.index:
        return g
    v = int(g.loc[RESIDUAL_TOPIC])
    res["caveats"].append(
        f"«{RESIDUAL_TOPIC}» ({num(v)}, {v / int(g.sum()) * 100:.1f}%) из рейтинга "
        f"исключено: это остаток маппинга тем, а не тема")
    return g.drop(index=RESIDUAL_TOPIC)


def caveats(df, vocab, query):
    """Оговорки считаются ИЗ ДАННЫХ, а не пишутся под конкретный вопрос."""
    out = []
    used = set(query.filters) | ({query.group_by} if query.group_by in df.columns else set())
    for c in sorted(used):
        f = vocab.fill.get(c, 1.0)
        if f < FILL_WARN:
            out.append(f"поле `{c}` заполнено на {f * 100:.1f}% — ответ только по "
                       f"заполненным строкам, остальные в счёт не идут")
    # Поле, заполненное лишь частью регионов, — это не «мало данных», а другая
    # выборка. Если доли внутри неё расходятся в разы, сводное число смешивает
    # несопоставимые величины, и это надо сказать до того, как его процитируют.
    if query.group_by == "sla_breach" or "sla_breach" in query.filters:
        sub = df[df.sla_breach.notna()]
        by_reg = sub.groupby("region").sla_breach.agg(["size", "mean"])
        if len(by_reg):
            names = ", ".join(f"{r} {m * 100:.1f}% ({num(n)})"
                              for r, (n, m) in by_reg.sort_values("mean").iterrows())
            out.append(f"поле заполнено только у {len(by_reg)} регионов из "
                       f"{df.region.nunique()}: {names}")
            lo, hi = by_reg["mean"].min(), by_reg["mean"].max()
            if lo > 0 and hi / lo >= 2:
                out.append(f"доли нарушения различаются в {hi / lo:.3f} раза при "
                           f"одинаковой схеме данных, и причина не установлена "
                           f"(раздел 3 CLAUDE.md). Сводное число по ним смешивает две "
                           f"несопоставимые величины — приводить его как общий уровень "
                           f"просрочки нельзя")
    if query.group_by == "region" or (not query.filters.get("region") and
                                      query.metric in ("top", "count", "compare")):
        out.append("объёмы регионов напрямую несопоставимы: регионы применяют разные "
                   "определения обращения, доля не-problem от 0.2% до 67.8% (раздел 5d)")
    if query.period:
        p = query.period
        covered, partial = [], []
        for r, (lo, hi) in vocab.windows.items():
            overlap = (min(hi, p.hi) - max(lo, p.lo)).total_seconds()
            full = (p.hi - p.lo).total_seconds()
            if overlap <= 0:
                continue
            (covered if overlap / full >= COVER_WARN else partial).append(r)
        n_in = len(covered) + len(partial)
        if n_in < len(vocab.windows):
            word = "региона" if 2 <= n_in % 10 <= 4 and not 12 <= n_in % 100 <= 14 else (
                "региона" if n_in % 10 == 1 and n_in % 100 != 11 else "регионов")
            out.append(f"в этот период попадают данные {n_in} {word} из "
                       f"{len(vocab.windows)}: у остальных выгрузка его не покрывает")
        # При «всей истории» частичное покрытие есть у всех по определению — это не
        # оговорка, а свойство вопроса; шумом она забивает настоящие предупреждения.
        if partial and p.label != "вся история":
            out.append("частично покрыт период у: " + ", ".join(sorted(partial)))
        elif p.label == "вся история" and len(vocab.windows) > 1:
            out.append("окна выгрузки у регионов разные — считается объединение всех "
                       "периодов, а не общий отрезок")
        for r, (n, a, b) in vocab.gaps.items():
            if r in covered + partial and a < p.hi and b >= p.lo:
                out.append(f"{r}: в выгрузке нет обращений {a:%Y-%m-%d} → {b:%Y-%m-%d} "
                           f"({n} дней) — число по этому региону занижено")
    return out


def run(df, vocab, query):
    d = apply_filters(df, query.filters)
    if query.period is not None:
        d = d[query.period.mask(d)]
    m, res = query.metric, {"query": query, "caveats": caveats(df, vocab, query)}

    if m == "count":
        res["number"] = len(d)
        res["answer"] = f"{num(len(d))} обращений"
        res["series"] = d.set_index("created_at").resample("MS").size()
    elif m == "compare":
        a = len(d)
        b = len(apply_filters(df, query.filters)[query.period_prev.mask(
            apply_filters(df, query.filters))])
        delta = (a - b) / b * 100 if b else float("nan")
        res["number"] = a
        res["answer"] = (f"{num(a)} против {num(b)} — "
                         f"{'больше' if a >= b else 'меньше'} на {abs(delta):.1f}%")
        res["table"] = pd.Series({query.period_prev.label: b, query.period.label: a})
    elif m == "top":
        g = d.groupby(query.group_by).size().sort_values(ascending=False)
        g = g[g.index.notna()]
        g = drop_residual(g, query, res)
        g = g.head(query.top)
        res["table"] = g
        res["number"] = int(g.iloc[0]) if len(g) else 0
        res["answer"] = ("; ".join(f"{k} — {num(v)}" for k, v in g.items())
                         if len(g) else "нет данных")
    elif m == "share":
        col = query.group_by
        if col == "status":
            s = d.status.dropna().str.lower()
            closed = s.str.contains("|".join(CLOSED_MARKS), na=False).sum()
            g = pd.Series({"закрыто": closed, "не закрыто": len(s) - closed})
        elif col == "sla_breach":
            s = d.sla_breach.dropna()
            g = pd.Series({"просрочено": int(s.sum()), "в срок": int((~s).sum())})
        else:
            g = d.groupby(col).size().sort_values(ascending=False)
            if col == "appeal_class":
                g.index = [CLASS_RU.get(i, i) for i in g.index]
        # Знаменатель берётся ДО исключения «прочего»: вопрос «доля всех обращений»
        # спрашивает про всё, и уменьшать целое под красивую долю нельзя.
        tot = int(g.sum())
        if col == "topic":
            g = drop_residual(g, query, res)
        if query.focus is not None:
            v = int(g.get(query.focus, 0))
            res["table"] = g
            res["number"] = v / tot * 100 if tot else float("nan")
            res["answer"] = (f"{query.focus} — {res['number']:.1f}% ({num(v)} из {num(tot)})"
                             if tot else "нет данных")
            return res
        res["table"] = g.head(query.top) if query.top else g
        if not tot:
            res["number"], res["answer"] = float("nan"), "нет данных"
        elif query.top:
            head = g.head(query.top)
            res["number"] = float(head.sum() / tot * 100)
            res["answer"] = (f"{res['number']:.1f}% — {', '.join(str(i) for i in head.index)} "
                             f"({num(head.sum())} из {num(tot)})")
        else:
            res["number"] = float(g.iloc[0] / tot * 100)
            res["answer"] = "; ".join(f"{k} — {v / tot * 100:.1f}% ({num(v)})"
                                      for k, v in g.items())
    elif m == "trend":
        s = d.set_index("created_at").resample("MS").size()
        res["series"] = s
        res["number"] = int(s.sum())
        res["answer"] = (f"от {num(s.min())} до {num(s.max())} в месяц, "
                         f"минимум {s.idxmin():%Y-%m}, максимум {s.idxmax():%Y-%m}")
    elif m == "growth":
        now = d.groupby("topic").size()
        was = apply_filters(df, query.filters)
        was = was[query.period_prev.mask(was)].groupby("topic").size()
        both = pd.concat([was.rename("было"), now.rename("стало")], axis=1).fillna(0)
        both = both[both["было"] >= GROWTH_MIN_BASE]
        both["прирост, %"] = (both["стало"] - both["было"]) / both["было"] * 100
        both = both.sort_values("прирост, %", ascending=False)
        if RESIDUAL_TOPIC in both.index:
            r0 = both.loc[RESIDUAL_TOPIC]
            res["caveats"].append(
                f"«{RESIDUAL_TOPIC}» из рейтинга исключено: это остаток маппинга тем, "
                f"а не тема. Его собственный прирост {r0['прирост, %']:+.1f}% "
                f"({num(r0['было'])} → {num(r0['стало'])}) — показатель качества "
                f"разметки, и его надо смотреть отдельно")
            both = both.drop(index=RESIDUAL_TOPIC)
        both = both.head(query.top)
        res["table"] = both
        res["number"] = float(both["прирост, %"].iloc[0]) if len(both) else float("nan")
        res["answer"] = (f"{both.index[0]} — {both['прирост, %'].iloc[0]:+.1f}% "
                         f"({num(both['было'].iloc[0])} → {num(both['стало'].iloc[0])})"
                         if len(both) else "нет данных")
        res["caveats"].append(
            f"темы с историей меньше {GROWTH_MIN_BASE} обращений в прошлом периоде "
            f"отброшены: процент прироста от малого числа не значит ничего. Порог "
            f"абсолютный и под длину периода не подстраивается — на коротком периоде "
            f"он строже, чем на длинном")
    elif m in ("dow", "weekend"):
        w = d.created_at.dt.dayofweek
        g = w.value_counts().sort_index()
        g.index = [DOW_RU[i] for i in g.index]
        res["table"] = g
        if m == "weekend":
            we = int(w.isin([5, 6]).sum())
            res["number"] = we / len(d) * 100 if len(d) else float("nan")
            res["answer"] = (f"{num(we)} из {num(len(d))} — {res['number']:.1f}% "
                             "приходится на субботу и воскресенье")
        else:
            res["number"] = int(g.max())
            res["answer"] = (f"больше всего в {g.idxmax()} ({num(g.max())}), "
                             f"меньше всего в {g.idxmin()} ({num(g.min())})")
    elif m == "peak_day":
        s = d.set_index("created_at").resample("D").size()
        res["series"] = s
        if not len(s):
            res["number"], res["answer"] = 0, "нет данных"
        else:
            day = s.idxmax()
            top = d[d.created_at.dt.normalize() == day].topic.value_counts()
            res["number"] = int(s.max())
            res["answer"] = (f"{day:%Y-%m-%d} — {num(s.max())} обращений"
                             + (f", крупнейшая тема «{top.index[0]}» ({num(top.iloc[0])})"
                                if len(top) else ""))
    elif m == "avg_day":
        g = d.groupby(d.created_at.dt.year).size()
        days = d.groupby(d.created_at.dt.year).created_at.agg(
            lambda x: (x.max().normalize() - x.min().normalize()).days + 1)
        avg = (g / days).round(1)
        nreg = d.groupby(d.created_at.dt.year).region.nunique()
        res["table"] = avg
        res["number"] = float(avg.iloc[-1]) if len(avg) else float("nan")
        res["answer"] = "; ".join(f"{y}: {v:.1f} в день" for y, v in avg.items())
        if nreg.nunique() > 1:
            res["caveats"].append(
                "число регионов в выгрузке по годам разное — "
                + ", ".join(f"{y}: {n}" for y, n in nreg.items())
                + ". Рост среднего в день частично объясняется тем, что регионов "
                  "становится больше, а не нагрузкой")
    elif m == "stopped":
        recent = query.period.hi - pd.DateOffset(months=STOPPED_MONTHS)
        was = d[d.created_at < recent].groupby("topic").size()
        now = d[d.created_at >= recent].groupby("topic").size()
        both = pd.concat([was.rename("до"), now.rename("после")], axis=1).fillna(0)
        dead = both[(both["до"] >= 100) & (both["после"] == 0)]
        res["table"] = both.sort_values("до", ascending=False)
        res["number"] = len(dead)
        res["answer"] = ("прекратились: " + ", ".join(f"{i} ({num(r['до'])})"
                                                      for i, r in dead.iterrows())
                         if len(dead) else "таких тем нет: все темы, по которым обращения "
                                           "были раньше, приходят и в последние 6 месяцев")
        res["caveats"].append(f"«прекратились» = ноль обращений с {recent:%Y-%m-%d} "
                              f"при 100+ обращениях до этой даты, в пределах периода")
    else:
        raise ValueError(m)
    return res


# ---------------------------------------------------------------- графики
def figure(res):
    """Одна plotly-фигура на ответ. От streamlit не зависит."""
    q = res["query"]
    title = q.text().splitlines()[0]
    if "series" in res and res["series"] is not None and len(res["series"]):
        s = res["series"]
        fig = px.bar(x=s.index, y=s.to_numpy(), labels={"x": "", "y": q.unit})
        if q.metric == "peak_day" and len(s):
            fig.add_annotation(x=s.idxmax(), y=float(s.max()), text="пик",
                               showarrow=True, arrowhead=2)
    elif "table" in res and res["table"] is not None and len(res["table"]):
        t = res["table"]
        if isinstance(t, pd.DataFrame):
            col = "прирост, %" if "прирост, %" in t.columns else t.columns[-1]
            fig = px.bar(x=t[col].to_numpy(), y=[str(i) for i in t.index],
                         orientation="h", labels={"x": col, "y": ""})
        else:
            fig = px.bar(x=t.to_numpy(), y=[str(i) for i in t.index],
                         orientation="h", labels={"x": q.unit, "y": ""})
        fig.update_yaxes(autorange="reversed")
    else:
        fig = go.Figure()
        fig.add_annotation(text="нет данных", showarrow=False)
    fig.update_layout(title=title, height=360, margin=dict(t=50, b=40))
    return fig


# ---------------------------------------------------------------- вывод
def answer(question, df=None, vocab=None):
    """Полный ответ на один вопрос: число, запрос, оговорки, фигура."""
    if df is None:
        df, vocab = load()
    q, why = parse(question, vocab, df)
    if q is None:
        return {"question": question, "refused": why}
    res = run(df, vocab, q)
    res["question"] = question
    res["figure"] = figure(res)
    return res


def render(res):
    out = [f"### {res['question']}", ""]
    if res.get("refused"):
        out += [f"**Ответить нельзя.** {res['refused']}", ""]
        return "\n".join(out)
    out += [f"**{res['answer']}**", ""]
    if res["query"].period:
        out += [f"Период: {res['query'].period.label}", ""]
    out += ["```sql", res["query"].text(), "```", ""]
    for c in res["caveats"]:
        out.append(f"> ⚠ {c}")
    if res["caveats"]:
        out.append("")
    return "\n".join(out)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="*", help="вопрос на естественном языке")
    a = ap.parse_args()
    if not a.question:
        ap.error("нужен вопрос: .venv/bin/python -m src.nlq \"сколько обращений в мае\"")
    df, vocab = load()
    print(render(answer(" ".join(a.question), df, vocab)))


if __name__ == "__main__":
    main()
