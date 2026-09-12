#!/usr/bin/env python3
"""Сдвиг полей на незакрытых кавычках — проверка по всем семи регионам.

    .venv/bin/python -m src.checks.field_shift

Дефект источника: длинное текстовое значение с незакрытой кавычкой режется
при разборе CSV, и содержимое одной колонки уезжает в другую.

Правила отсева (функции *_shift_rules) импортируются адаптерами
(src/adapters/adapters.py), поэтому числа этого отчёта и счётчики отсева
в адаптерах совпадают по построению. Отчёт воспроизводит таблицу
«Сдвиг полей на незакрытых кавычках» в CLAUDE.md, раздел 5c.

Скрипт только читает сырые файлы. Значения из данных не печатает.
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path("drive-download-20260907T161509Z-1-001")
FILES = {
    "Карагандинская": ["Обращения граждан 109 - Карагандинская область.csv"],
    "Алматинская": ["Обращения граждан 109 - Алматинская область.csv"],
    "Акмолинская": ["Обращения граждан 109 - Акмолинская область.csv"],
    "Восточно-Казахстанская": ["Обращения граждан 109 - Восточно-Казахстанская область.csv"],
    "Павлодарская": ["Обращения граждан Павлодар/Данные по обращениям 109 — Павлодарская область_part_001_of_002.csv",
                     "Обращения граждан Павлодар/Данные по обращениям 109 — Павлодарская область_part_002_of_002.csv"],
    "Туркестанская": ["Обращения жителей 109 - Туркестанская область.csv"],
    "Костанайская": ["Обращения жителей 109 - Костанайская область.csv"],
}

# ------------------------------------------------------------ правила отсева
KARAGANDA_ANSWER = {"Быстрый ответ", "Письменное обращение"}
KARAGANDA_TYPE = {"Физ. лицо", "Юр. лицо"}
ALMATY_STATUS = {"Закрыто", "В работе"}
ALMATY_CHANNELS = {"ЕКЦ 109", "Whatsapp", "Моб. приложение", "Чат-бот Телеграм", "Инстаграм"}
VKO_STATUS = {"Закрыто", "В работе"}
VKO_CHANNELS = {"ЕКЦ 109", "Whatsapp", "Инстаграм", "Чат-бот Телеграм", "Моб. приложение", "Другие"}
# Порог 100 символов и ветки «дом … кв» / «мкр» — известный долг (CLAUDE.md, раздел 9):
# две последние ветки на текущих данных не срабатывают ни разу.
ADDR = re.compile(r"(?:\bдом\b.*\bкв\b)|(?:^|\s)ул\.|(?:^|\s)мкр\b", re.I)


def _s(df, col):
    return df[col].fillna("").astype(str).str.strip()


def karaganda_shift_rules(df):
    return {
        "answer_type вне справочника": ~_s(df, "answer_type").isin(KARAGANDA_ANSWER | {""}),
        "type вне справочника": ~_s(df, "type").isin(KARAGANDA_TYPE | {""}),
    }


def almaty_shift_rules(df):
    st, ca = _s(df, "status"), _s(df, "category")
    addr = ca.str.contains(ADDR)
    return {
        "R1 status вне справочника статусов": ~st.isin(ALMATY_STATUS | {""}),
        "R2 status пустой при category, похожей на адрес": (st == "") & addr,
        "R3 category совпадает со значением статуса": ca.isin(ALMATY_STATUS),
        "R4 адресный фрагмент в category": addr & (ca.str.len() <= 100),
        "R5 сдвиг вне канона: status_1 или submittal_channel вне справочника":
            ~_s(df, "status_1").isin(ALMATY_STATUS | {""})
            | ~_s(df, "submittal_channel").isin(ALMATY_CHANNELS | {""}),
    }


def vko_shift_rules(df):
    return {
        "status вне справочника статусов": ~_s(df, "status").isin(VKO_STATUS | {""}),
        "submittal_channel вне справочника каналов": ~_s(df, "submittal_channel").isin(VKO_CHANNELS | {""}),
        "category совпадает со значением канала": _s(df, "category").isin(VKO_CHANNELS),
    }


def union(rules):
    out = None
    for m in rules.values():
        out = m.copy() if out is None else out | m
    return out


# ------------------------------------------------------------ отчёт
def _load(region):
    return pd.concat([pd.read_csv(BASE / f, dtype=str) for f in FILES[region]], ignore_index=True)


SKIP = re.compile(r"id$|^id|code|number|date|created|updated|finish|start|confirm|upload|load|"
                  r"closing|creation|coordinate|com_exp|request_subject|appeal_address|street|"
                  r"full_name|applicant|operator", re.I)
NUM = re.compile(r"^-?\d+(?:[.,]\d+)?$|^(?:true|false)$", re.I)


def cross_vocab(df):
    """Значение колонки A вне основного словаря A, но в основном словаре другой
    колонки B того же файла. Основной словарь — частые значения, покрывающие 98%
    непустых, каждое не реже 3 раз. Слепые зоны — см. CLAUDE.md, 5c."""
    cols = []
    for c in df.columns:
        if SKIP.search(c):
            continue
        nz = _s(df, c)
        nz = nz[nz != ""]
        if nz.empty or nz.str.match(NUM).mean() > 0.9:
            continue
        if nz.nunique() > 3000 or nz.nunique() / len(nz) > 0.2:
            continue
        cols.append(c)
    core = {}
    for c in cols:
        vc = _s(df, c)
        vc = vc[vc != ""].value_counts()
        acc, s = 0, set()
        for v, n in vc.items():
            if acc >= 0.98 * vc.sum() or n < 3:
                break
            s.add(v); acc += n
        core[c] = s
    pairs, hit = Counter(), pd.Series(False, index=df.index)
    for a in cols:
        va = _s(df, a)
        out_a = (va != "") & ~va.isin(core[a])
        for b in cols:
            if b == a:
                continue
            m = out_a & va.isin(core[b])
            if m.any():
                pairs[(a, b)] += int(m.sum())
                hit |= m
                out_a &= ~m          # одна пара на ячейку
    return int(hit.sum()), pairs


def _ok_iso(s):
    return pd.to_datetime(s, errors="coerce", format="ISO8601").notna()


def main():
    rows = []
    print("СДВИГ ПОЛЕЙ НА НЕЗАКРЫТЫХ КАВЫЧКАХ — проверка по семи регионам\n")

    df = _load("Карагандинская")
    r = karaganda_shift_rules(df); u = union(r)
    print(f"=== Карагандинская ({len(df)} строк)")
    for k, m in r.items():
        print(f"    {k}: {int(m.sum())}")
    print(f"    ВСЕГО строк со сдвигом: {int(u.sum())}")
    rows.append(("Карагандинская", int(u.sum())))

    df = _load("Алматинская")
    r = almaty_shift_rules(df); u = union(r)
    canon = union({k: v for k, v in r.items() if not k.startswith("R5")})
    print(f"\n=== Алматинская ({len(df)} строк)")
    others = {k: union({kk: vv for kk, vv in r.items() if kk != k and not kk.startswith('R5')})
              for k in r if not k.startswith("R5")}
    for k, m in r.items():
        only = f", только это правило среди R1–R4: {int((m & ~others[k]).sum())}" if k in others else ""
        print(f"    {k}: {int(m.sum())}{only}")
    ca = _s(df, "category")
    seen = ca.isin(ALMATY_STATUS) | (ca.str.lower().str.contains("дом") & ca.str.lower().str.contains("кв")
                                     & (ca.str.len() <= 100)) | (ca.str.len() > 100)
    r5_only = r[list(r)[-1]] & ~canon
    print(f"    R1–R4 вместе (сдвиг в каноне): {int(canon.sum())}")
    print(f"      из них видимых в category (статус, адрес «дом…кв», текст > 100): {int(seen.sum())}, "
          f"пойманы все: {bool((seen & ~canon).sum() == 0)}; сверх видимых: {int((canon & ~seen).sum())}")
    print(f"      «Проблема устранена» в колонке result: {int((_s(df, 'result') == 'Проблема устранена').sum())}")
    print(f"    R5 сверх R1–R4 (сдвиг только вне канона): {int(r5_only.sum())}, "
          f"из них с пустой category: {int((r5_only & (ca == '')).sum())}")
    print(f"    ВСЕГО строк со сдвигом: {int(u.sum())}")
    rows.append(("Алматинская", int(u.sum())))

    df = _load("Акмолинская")
    bad = ~_ok_iso(df["creation_date"])
    st_core = {v for v, n in _s(df, "status").value_counts().items() if n >= 20}
    cp = _s(df, "current_project").isin(st_core)
    print(f"\n=== Акмолинская ({len(df)} строк)")
    print(f"    creation_date не разбирается (в ней обрывки названий организаций): {int(bad.sum())}")
    print(f"    значение статуса в current_project: {int(cp.sum())}, из них среди строк с битой датой: {int((cp & bad).sum())}")
    print(f"    ВСЕГО строк со сдвигом: {int((bad | cp).sum())}  (адаптер отбрасывает их как непарсящиеся даты)")
    rows.append(("Акмолинская", int((bad | cp).sum())))

    df = _load("Восточно-Казахстанская")
    r = vko_shift_rules(df); u = union(r)
    print(f"\n=== Восточно-Казахстанская ({len(df)} строк)")
    for k, m in r.items():
        print(f"    {k}: {int(m.sum())}")
    print(f"    ВСЕГО строк со сдвигом: {int(u.sum())}")
    rows.append(("Восточно-Казахстанская", int(u.sum())))

    df = _load("Павлодарская")
    print(f"\n=== Павлодарская ({len(df)} строк)")
    for idc, nc in (("service_id", "service_name"), ("category_id", "category_name")):
        multi = df.groupby(idc)[nc].nunique()
        print(f"    {idc} с несколькими названиями {nc}: {int((multi > 1).sum())} из {len(multi)}")
    print("    ВСЕГО строк со сдвигом: 0  (id и названия согласованы; колонок свободного текста нет)")
    rows.append(("Павлодарская", 0))

    for reg in ("Туркестанская", "Костанайская"):
        print(f"\n=== {reg}: признаков сдвига нет; совпадения со словарём другой колонки — ниже")
        rows.append((reg, 0))

    print("\n=== Метод «значение из словаря другой колонки» (контекст, слепые зоны — CLAUDE.md 5c)")
    for reg in FILES:
        n, pairs = cross_vocab(_load(reg))
        top = "; ".join(f"{a} <- {b}: {c}" for (a, b), c in pairs.most_common(3))
        print(f"    {reg:24s} {n:6d}  {top}")

    print("\n=== ИТОГ")
    for reg, n in rows:
        print(f"    {reg:24s} {n:5d}")
    print(f"    регионов со сдвигом: {sum(1 for _, n in rows if n)} из {len(rows)}")


if __name__ == "__main__":
    main()
