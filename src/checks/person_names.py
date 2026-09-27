"""Названия, которые по форме организация, а по содержанию человек (раздел 1).

Проверка канона 5c искала только цифры (телефоны, ИИН), проверка синтетики
(src/synth/checks.py) — имена в связном тексте. Обе пропускали класс, найденный
в 5o: «ИП Фамилия», «личное имя (…)», крестьянское хозяйство и ТОО на фамилию.
Такое название — ФИО человека, и печатать его можно только маской.

    person_hits(name) -> список причин, пустой — не человек
    safe(name)        -> «ИП-3», «ПКСК-1»…, если это человек; иначе как есть

Маска — стабильный номер на значение, а не звёздочки: шесть ИП Костаная под
звёздочками выглядели бы одинаково, и в таблице риска получилось бы несколько
строк с одним названием. Номер берётся из реестра всех таких названий канона и
теста модели, отсортированного по значению, — один исполнитель под одним номером
в демо, витрине, NLQ и отчётах 5o. Маскируется только вывод: в данных названия
остаются как есть, чтобы расчёты не склеили разные службы.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

from src import paths
from src.synth.checks import ALLOW, NAME_RE, SURNAME

# Формы, которые всегда означают человека: индивидуальный предприниматель,
# крестьянское (фермерское) хозяйство — по-русски и по-казахски.
PERSON_FORM = re.compile(
    r"(?<![\wА-Яа-яЁё])(ИП|И\.П\.|ЖК|ШҚ|ШК|КХ|К/Х|К\.Х\.|ФХ|Ф/Х|"
    r"индивидуальн\w* предприним\w*|жеке кәсіпкер\w*|"
    r"крестьянск\w* (\(фермерск\w*\) )?хозяйств\w*|фермерск\w* хозяйств\w*|"
    r"шаруа қожалығ\w*)(?![\wА-Яа-яЁё])", re.I)
PERSON_BEFORE_QUOTE = re.compile(r"(?<![\wА-Яа-яЁё])(\S{2,5})\s*[«\"“]")
# Организационные формы юрлиц: название в кавычках после них может быть фамилией.
ORG_FORM = re.compile(r"(?<![\wА-Яа-яЁё])(ТОО|ЖШС|АО|АҚ|ООО|ГКП|МКК|КГП|КГУ|ГУ|РГП|ПК|ОЮЛ)"
                      r"(?![\wА-Яа-яЁё])")
QUOTED = re.compile(r"[«\"“]([^»\"”]{2,60})[»\"”]")
INITIALS = re.compile(r"\b[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]{2,}\s+[А-ЯЁӘҒҚҢӨҰҮҺІ]\.\s?"
                      r"([А-ЯЁӘҒҚҢӨҰҮҺІ]\.)?|\b[А-ЯЁӘҒҚҢӨҰҮҺІ]\.\s?[А-ЯЁӘҒҚҢӨҰҮҺІ]\.\s?"
                      r"[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]{2,}")
# Топонимы с окончанием фамилии: «Карагандинский», «Костанайская», «Павлодарская».
PLACE_ADJ = re.compile(r"(ский|ская|ское|ских|ского|ской|ском)$", re.I)
SURNAME_STRICT = re.compile(r"^[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]{2,}"
                            r"(ов|ова|ев|ева|ин|ина|енко|баев|баева|ұлы|қызы|улы|кызы)$")


def _mask_word(w: str) -> str:
    return w[:2] + "*" * max(len(w) - 2, 0)


def person_hits(name) -> list[str]:
    if not isinstance(name, str) or not name.strip():
        return []
    s = name.strip()
    hits = []
    # «КХ» внутри названия организации чаще «коммунальное хозяйство» («Отдел КХ,
    # пассажирского транспорта…» — 7 470 строк Караганды), «ИП» в хвосте названия
    # департамента — не форма собственности. Форма человека — только в начале
    # названия или перед названием в кавычках.
    lead = PERSON_FORM.match(s.lstrip("«\"“ "))
    quoted = any(PERSON_FORM.fullmatch(m.group(1)) for m in PERSON_BEFORE_QUOTE.finditer(s))
    if lead or quoted:
        hits.append("ИП / крестьянское хозяйство — всегда человек")
    if INITIALS.search(s):
        hits.append("фамилия с инициалами")
    first = re.split(r"[\s(«\"]", s, maxsplit=1)[0]
    m = NAME_RE.fullmatch(first) if first else None
    if m and first.lower() not in ALLOW and not ORG_FORM.fullmatch(first):
        hits.append("начинается с личного имени")
    if ORG_FORM.search(s):
        for q in QUOTED.findall(s):
            words = [w for w in re.split(r"[\s\-]+", q.strip()) if w]
            if not 1 <= len(words) <= 3:
                continue
            for w in words:
                wl = w.lower()
                if wl in ALLOW or PLACE_ADJ.search(wl):
                    continue
                if SURNAME_STRICT.match(w) or NAME_RE.fullmatch(w):
                    # требует проверки: имена часто совпадают с названиями городов
                    hits.append("юрлицо, названное фамилией или именем — проверить")
                    break
    return hits


UNIFIED = paths.UNIFIED
PRED = paths.PREDICTIONS


def _key(name) -> str:
    return " ".join(str(name).split())


def _kind(name: str) -> str:
    """Форма для подписи: ИП / КХ / ПКСК / ТОО…; иначе «исполнитель»."""
    s = name.strip().lstrip("«\"“ ")
    m = PERSON_FORM.match(s)
    if m:
        f = m.group(1).upper().replace(".", "").replace("/", "")
        return "ИП" if f.startswith(("ИП", "ИНДИВИДУАЛ", "ЖЕКЕ")) else "КХ"
    m = re.search(r"\((П?КСК)\b", s)
    if m:
        return m.group(1)
    m = ORG_FORM.search(s)
    return m.group(1) if m else "исполнитель"


@lru_cache(maxsize=1)
def registry() -> dict[str, str]:
    """Все названия-люди канона и теста модели -> стабильная подпись «ИП-1»…"""
    import pandas as pd
    vals = set()
    if UNIFIED.exists():
        vals |= set(pd.read_parquet(UNIFIED, columns=["executor"]).executor.dropna().map(_key))
    if PRED.exists():
        vals |= set(pd.read_csv(PRED, usecols=["category"]).category.dropna().map(_key))
    out, n = {}, {}
    for v in sorted(v for v in vals if person_hits(v)):
        k = _kind(v)
        n[k] = n.get(k, 0) + 1
        out[v] = f"{k}-{n[k]}"
    return out


def safe(name):
    """Подпись вместо названия, если это человек; иначе название как есть.
    Значение вне реестра (новая выгрузка) — форма и 4 знака хеша: тоже стабильно."""
    if not person_hits(name):
        return name
    k = _key(name)
    lab = registry().get(k)
    return lab or f"{_kind(k)}-{hashlib.sha256(k.encode()).hexdigest()[:4]}"
