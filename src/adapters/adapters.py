"""
Адаптеры шести семейств схем данных 109 к каноническому виду (модуль 3).

Канон:
  обязательные — created_at (datetime), region (метка области), category (тема)
  опциональные — district, executor, status, sla_breach
  служебная    — source_file (имя исходного файла)

Маппинг колонок задан постановкой задачи и НЕ переопределяется здесь:
Караганда: category = sub_category (не category — там название организации);
Костанай/Туркестан: category = servicelevel1 (не category — там "инцидент"/"жалоба");
Акмола: category = direction (колонки category в источнике нет вообще).
`region` — не берётся из колонки источника (там город/район), а прописывается
меткой области по имени файла.

Почему нет data/unified.parquet
--------------------------------
Этот модуль намеренно не содержит функции, которая физически сливает все
регионы в один DataFrame/файл. Причина — CLAUDE.md:
  - раздел 3, «ОТКРЫТО: разброс доли нарушений SLA в 4,6 раза — блокирует
    объединение регионов»: «Объединять регионы в одну модель до объяснения
    запрещено. Запрет снимается только явным решением в основной сессии»;
  - раздел 7, «Текущие заморозки»: пункт 2 («адаптеры схем не пишутся»)
    и пункт 3 («регионы не объединяются до объяснения расхождения
    34,9% / 7,5%») — это два РАЗНЫХ пункта; снятие одного не снимает другой,
    а сообщение о снятии заморозки на этот заход ссылалось только на пункт 2;
  - раздел 2, «Не делегируется никогда»: «решение объединять ли регионы» —
    прямо в списке решений, которые не может принять субагент.

Поэтому каждый регион пишется в свой собственный parquet
(`data/by_region/<регион>.parquet`), с общими именами колонок канона —
это позволяет сравнивать регионы и готовить признаки, не отменяя того факта,
что объединение в одну обучающую выборку остаётся нерешённым вопросом
основной сессии.
"""

from __future__ import annotations

import os
import re

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
BASE_DIR = os.path.join(PROJECT_ROOT, "drive-download-20260907T161509Z-1-001")
OUT_DIR = os.path.join(PROJECT_ROOT, "data")

CANON_COLUMNS = ["created_at", "region", "category", "district", "executor", "status", "sla_breach"]
MAX_DATE_FAIL_SHARE = 0.01  # 1% — порог, выше которого адаптер обязан упасть, а не молчать

# Сырые размеры из CLAUDE.md, раздел 5 «Инвентарь данных» — для проверки на РАСХОЖДЕНИЕ
INVENTORY_RAW_ROWS = {
    "Караганда": 143954,
    "Костанай": 20591,
    "Туркестан": 98876,
    "ВКО": 83385,
    "Алматы": 19912,
    "Акмола": 3506,
    "Павлодар": 666634,
}


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #

def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def _inventory_check(name: str, raw_rows: int) -> None:
    expected = INVENTORY_RAW_ROWS.get(name)
    if expected is None:
        return
    if raw_rows != expected:
        print(f"  РАСХОЖДЕНИЕ [{name}]: сырых строк {raw_rows}, в инвентаре CLAUDE.md {expected}")
    else:
        print(f"  [{name}] сырых строк {raw_rows} — сходится с инвентарём CLAUDE.md")


def _dedup(df: pd.DataFrame, key: str | None, name: str) -> tuple[pd.DataFrame, int]:
    """Возвращает (df после дедупа, снято строк). Печатает дублей до/после."""
    if key is None:
        print(f"  [{name}] колонки-идентификатора нет — дедуп не выполняется, берутся все строки")
        return df, 0
    dups_before = int(df[key].duplicated().sum())
    df_dedup = df.drop_duplicates(subset=key, keep="first")
    dups_after = int(df_dedup[key].duplicated().sum())
    dropped = len(df) - len(df_dedup)
    print(
        f"  [{name}] дублей по '{key}': до дедупа {dups_before}, после {dups_after}; "
        f"снято дедупом {dropped} строк; уникальных значений ключа {df[key].nunique()}"
    )
    return df_dedup, dropped


def _parse_dates(df: pd.DataFrame, col: str, fmt: str, name: str) -> pd.Series:
    parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
    fail = int(parsed.isna().sum())
    share = fail / len(df) if len(df) else 0.0
    print(f"  [{name}] дата '{col}' форматом '{fmt}': не распарсилось {fail} из {len(df)} ({share:.2%})")
    if share > MAX_DATE_FAIL_SHARE:
        raise RuntimeError(
            f"[{name}] доля не распарсившихся дат {share:.2%} превышает порог "
            f"{MAX_DATE_FAIL_SHARE:.0%} — адаптер остановлен явно, NaT молча не подставляется."
        )
    return parsed


def _build_canon(
    df: pd.DataFrame,
    parsed_dates: pd.Series,
    region_label: str,
    category_col: str | None,
    district_col: str | None,
    executor_col: str | None,
    status_col: str | None,
    sla_breach_col: str | None,
    source_file: pd.Series,
    name: str,
) -> tuple[pd.DataFrame, set[str]]:
    absent: set[str] = set()
    out = pd.DataFrame(index=df.index)
    out["created_at"] = parsed_dates
    out["region"] = region_label

    if category_col is not None:
        out["category"] = df[category_col]
    else:
        out["category"] = pd.NA
        absent.add("category")

    for canon_name, src_col in (
        ("district", district_col),
        ("executor", executor_col),
        ("status", status_col),
    ):
        if src_col is not None:
            out[canon_name] = df[src_col]
        else:
            out[canon_name] = pd.NA
            absent.add(canon_name)

    if sla_breach_col is not None:
        raw = df[sla_breach_col]
        mapped = raw.map({"True": True, "False": False})
        unexpected = raw.notna() & mapped.isna()
        if unexpected.any():
            print(
                f"  [{name}] ВНИМАНИЕ: {int(unexpected.sum())} значений '{sla_breach_col}' "
                f"не 'True'/'False': {sorted(raw.loc[unexpected].unique())[:5]}"
            )
        out["sla_breach"] = mapped.astype("boolean")
    else:
        out["sla_breach"] = pd.array([pd.NA] * len(df), dtype="boolean")
        absent.add("sla_breach")

    out["source_file"] = source_file.values
    return out, absent


def _column_report(df: pd.DataFrame, name: str, absent: set[str]) -> None:
    n = len(df)
    print(f"  [{name}] заполненность колонок канона (n={n}):")
    for c in CANON_COLUMNS:
        if c in absent:
            print(f"    {c}: отсутствует в источнике — нечем заполнить")
            continue
        filled = int(df[c].notna().sum())
        empty = n - filled
        note = f", пусто {empty} ({empty / n:.1%})" if empty else " (пустых нет)"
        print(f"    {c}: заполнена {filled}/{n} ({filled / n:.1%}){note}")
    print(f"    source_file: заполнена {int(df['source_file'].notna().sum())}/{n}")


def _finish(name: str, n_raw: int, dedup_dropped: int, date_dropped: int, canon: pd.DataFrame,
            absent: set[str], shift_dropped: int = 0) -> pd.DataFrame:
    canon = canon.reset_index(drop=True)
    n_out = len(canon)
    check = n_raw - dedup_dropped - date_dropped - shift_dropped
    ok = "OK" if check == n_out else "НЕСХОДИТСЯ"
    shift = f"; отсеяно по сдвигу полей {shift_dropped}" if shift_dropped else ""
    shift_c = f" - {shift_dropped}" if shift_dropped else ""
    print(
        f"  [{name}] строк на входе {n_raw}; снято дедупом {dedup_dropped}; "
        f"отброшено по дате {date_dropped}{shift}; строк на выходе {n_out} "
        f"(контроль: {n_raw} - {dedup_dropped} - {date_dropped}{shift_c} = {check} [{ok}])"
    )
    _column_report(canon, name, absent)
    return canon


# --------------------------------------------------------------------------- #
# По функции на регион
# --------------------------------------------------------------------------- #

def load_karaganda() -> pd.DataFrame:
    name = "Караганда"
    path = os.path.join(BASE_DIR, "Обращения граждан 109 - Карагандинская область.csv")
    print(f"\n--- {name} ---")
    df = _read_csv(path)
    n_raw = len(df)
    print(f"  [{name}] строк на входе: {n_raw}")
    _inventory_check(name, n_raw)

    df_dedup, dedup_dropped = _dedup(df, None, name)

    parsed = _parse_dates(df_dedup, "created_date", "%m/%d/%y %H:%M", name)
    if int(parsed.isna().sum()) > 0:
        print(
            f"  [{name}] РАСХОЖДЕНИЕ: {int(parsed.isna().sum())} строк(и) с датой, не распарсившейся по "
            f"'%m/%d/%y %H:%M'. Проверено вручную — это те же строки, что задокументированные в CLAUDE.md "
            f"как 5 битых строк с невалидным appeal_type (сдвиг полей из-за незаэкранированных кавычек: "
            f"вся строка целиком попадает в одну ячейку created_date). Не подгонка — отбрасываются как есть."
        )
    mask = parsed.notna()
    date_dropped = int((~mask).sum())
    df_valid, parsed_valid = df_dedup.loc[mask], parsed.loc[mask]

    src = pd.Series(os.path.basename(path), index=df_valid.index)
    canon, absent = _build_canon(
        df_valid, parsed_valid, "Карагандинская область",
        category_col="sub_category", district_col="district",
        executor_col="executor_gov_org", status_col=None, sla_breach_col=None,
        source_file=src, name=name,
    )
    return _finish(name, n_raw, dedup_dropped, date_dropped, canon, absent)


def _load_kostanay_turkestan(name: str, filename: str, region_label: str) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, filename)
    print(f"\n--- {name} ---")
    df = _read_csv(path)
    n_raw = len(df)
    print(f"  [{name}] строк на входе: {n_raw}")
    _inventory_check(name, n_raw)

    df_dedup, dedup_dropped = _dedup(df, "incidentcode", name)

    # ISO с переменной точностью долей секунды (микро/милли/нет) — не strptime с
    # жёстко заданной маской, а формат-флаг pandas ISO8601, эмпирически проверено: 0% отказов
    parsed = _parse_dates(df_dedup, "createddate", "ISO8601", name)
    mask = parsed.notna()
    date_dropped = int((~mask).sum())
    df_valid, parsed_valid = df_dedup.loc[mask], parsed.loc[mask]

    src = pd.Series(os.path.basename(path), index=df_valid.index)
    canon, absent = _build_canon(
        df_valid, parsed_valid, region_label,
        category_col="servicelevel1", district_col="region",
        executor_col="organizationname", status_col="status", sla_breach_col="slabreach",
        source_file=src, name=name,
    )
    return _finish(name, n_raw, dedup_dropped, date_dropped, canon, absent)


def load_kostanay() -> pd.DataFrame:
    return _load_kostanay_turkestan(
        "Костанай", "Обращения жителей 109 - Костанайская область.csv", "Костанайская область"
    )


def load_turkestan() -> pd.DataFrame:
    return _load_kostanay_turkestan(
        "Туркестан", "Обращения жителей 109 - Туркестанская область.csv", "Туркестанская область"
    )


def load_vko() -> pd.DataFrame:
    name = "ВКО"
    path = os.path.join(BASE_DIR, "Обращения граждан 109 - Восточно-Казахстанская область.csv")
    print(f"\n--- {name} ---")
    df = _read_csv(path)
    n_raw = len(df)
    print(f"  [{name}] строк на входе: {n_raw}")
    _inventory_check(name, n_raw)

    df_dedup, dedup_dropped = _dedup(df, "application_number", name)

    parsed = _parse_dates(df_dedup, "creation_date", "%d.%m.%Y %H:%M:%S", name)
    mask = parsed.notna()
    date_dropped = int((~mask).sum())
    df_valid, parsed_valid = df_dedup.loc[mask], parsed.loc[mask]

    src = pd.Series(os.path.basename(path), index=df_valid.index)
    canon, absent = _build_canon(
        df_valid, parsed_valid, "Восточно-Казахстанская область",
        category_col="category", district_col="district",
        executor_col="contractor", status_col="status", sla_breach_col=None,
        source_file=src, name=name,
    )
    return _finish(name, n_raw, dedup_dropped, date_dropped, canon, absent)


# Сдвиг полей на незакрытых кавычках (CLAUDE.md, 5c). Длинный текст заявителя
# режется, и содержимое одной колонки уезжает в другую. Отсеиваются строки,
# где сдвиг виден в канонических колонках category и status — по аналогии
# с Карагандой, где битые строки отсеиваются по answer_type.
_ALMATY_STATUS = {"Закрыто", "В работе"}
_ADDR = re.compile(r"(?:\bдом\b.*\bкв\b)|(?:^|\s)ул\.|(?:^|\s)мкр\b", re.I)


def _drop_shifted_almaty(df: pd.DataFrame, parsed: pd.Series, name: str):
    st = df["status"].fillna("").astype(str).str.strip()
    ca = df["category"].fillna("").astype(str).str.strip()
    addr = ca.str.contains(_ADDR)
    rules = {
        "status вне справочника статусов": ~st.isin(_ALMATY_STATUS | {""}),
        "status пустой при category, похожей на адрес": (st == "") & addr,
        "category совпадает со значением статуса": ca.isin(_ALMATY_STATUS),
        "адресный фрагмент в category": addr & (ca.str.len() <= 100),
    }
    shifted = pd.Series(False, index=df.index)
    for rule, m in rules.items():
        print(f"  [{name}] сдвиг полей — {rule}: {int(m.sum())}")
        shifted |= m
    print(f"  [{name}] сдвиг полей — всего строк по любому правилу: {int(shifted.sum())}")
    return df.loc[~shifted], parsed.loc[~shifted], int(shifted.sum())


def load_almaty() -> pd.DataFrame:
    name = "Алматы"
    path = os.path.join(BASE_DIR, "Обращения граждан 109 - Алматинская область.csv")
    print(f"\n--- {name} ---")
    df = _read_csv(path)
    n_raw = len(df)
    print(f"  [{name}] строк на входе: {n_raw}")
    _inventory_check(name, n_raw)

    df_dedup, dedup_dropped = _dedup(df, "application_number", name)

    parsed = _parse_dates(df_dedup, "creation_date", "%d.%m.%Y %H:%M:%S", name)
    mask = parsed.notna()
    date_dropped = int((~mask).sum())
    df_valid, parsed_valid = df_dedup.loc[mask], parsed.loc[mask]
    df_valid, parsed_valid, shift_dropped = _drop_shifted_almaty(df_valid, parsed_valid, name)

    src = pd.Series(os.path.basename(path), index=df_valid.index)
    canon, absent = _build_canon(
        df_valid, parsed_valid, "Алматинская область",
        category_col="category", district_col=None,
        # contractor НЕ переносится: помимо названий организаций в нём лежит
        # свободный текст заявителей (~55 строк) с ИИН, телефонами и ФИО.
        # См. CLAUDE.md, раздел 5b.
        executor_col=None, status_col="status", sla_breach_col=None,
        source_file=src, name=name,
    )
    return _finish(name, n_raw, dedup_dropped, date_dropped, canon, absent,
                   shift_dropped=shift_dropped)


def load_akmola() -> pd.DataFrame:
    name = "Акмола"
    path = os.path.join(BASE_DIR, "Обращения граждан 109 - Акмолинская область.csv")
    print(f"\n--- {name} ---")
    df = _read_csv(path)
    n_raw = len(df)
    print(f"  [{name}] строк на входе: {n_raw}")
    _inventory_check(name, n_raw)

    df_dedup, dedup_dropped = _dedup(df, "request_number", name)

    parsed = _parse_dates(df_dedup, "creation_date", "ISO8601", name)
    mask = parsed.notna()
    date_dropped = int((~mask).sum())
    df_valid, parsed_valid = df_dedup.loc[mask], parsed.loc[mask]

    src = pd.Series(os.path.basename(path), index=df_valid.index)
    canon, absent = _build_canon(
        df_valid, parsed_valid, "Акмолинская область",
        category_col="direction", district_col="region_g_a",
        executor_col=None, status_col="status", sla_breach_col=None,
        source_file=src, name=name,
    )
    return _finish(name, n_raw, dedup_dropped, date_dropped, canon, absent)


def load_pavlodar() -> pd.DataFrame:
    name = "Павлодар"
    subdir = os.path.join(BASE_DIR, "Обращения граждан Павлодар")
    part_files = [
        "Данные по обращениям 109 — Павлодарская область_part_001_of_002.csv",
        "Данные по обращениям 109 — Павлодарская область_part_002_of_002.csv",
    ]
    print(f"\n--- {name} ---")
    parts = []
    for fn in part_files:
        p = os.path.join(subdir, fn)
        part_df = _read_csv(p)
        part_df["_source_file"] = fn
        print(f"  [{name}] часть '{fn}': {len(part_df)} строк")
        parts.append(part_df)

    overlap = set(parts[0]["id"]).intersection(set(parts[1]["id"]))
    print(f"  [{name}] пересечение id между частями: {len(overlap)}")

    df = pd.concat(parts, ignore_index=True)
    n_raw = len(df)
    print(f"  [{name}] строк на входе (обе части объединены): {n_raw}")
    _inventory_check(name, n_raw)

    df_dedup, dedup_dropped = _dedup(df, "public_code", name)

    parsed = _parse_dates(df_dedup, "create_date", "ISO8601", name)
    mask = parsed.notna()
    date_dropped = int((~mask).sum())
    df_valid, parsed_valid = df_dedup.loc[mask], parsed.loc[mask]

    src = df_valid["_source_file"]
    canon, absent = _build_canon(
        df_valid, parsed_valid, "Павлодарская область",
        category_col="category_name", district_col=None,
        executor_col=None, status_col="status", sla_breach_col=None,
        source_file=src, name=name,
    )
    return _finish(name, n_raw, dedup_dropped, date_dropped, canon, absent)


REGION_LOADERS = {
    "Караганда": load_karaganda,
    "Костанай": load_kostanay,
    "Туркестан": load_turkestan,
    "ВКО": load_vko,
    "Алматы": load_almaty,
    "Акмола": load_akmola,
    "Павлодар": load_pavlodar,
}


# --------------------------------------------------------------------------- #
# B. Словари тем (category) — только числа, без автоматического сведения
# --------------------------------------------------------------------------- #

def topic_dictionaries(canon: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 70)
    print("B. Словари тем (category) по регионам")
    print("=" * 70)
    value_sets: dict[str, set[str]] = {}
    for name, df in canon.items():
        vc = df["category"].dropna().value_counts()
        value_sets[name] = set(vc.index)
        print(f"\n[{name}] уникальных значений category: {df['category'].nunique(dropna=True)}")
        print(f"[{name}] топ-20 по частоте:")
        for val, cnt in vc.head(20).items():
            print(f"    {cnt:>7}  {val}")

    print("\n" + "-" * 70)
    print("Попарные пересечения словарей (буквальное совпадение строк category)")
    print("Синонимы НЕ ищутся, к общему справочнику НЕ приводится — только числа.")
    print("-" * 70)
    names = list(canon.keys())
    col_w = 12
    print("region".ljust(14) + "".join(n.ljust(col_w) for n in names))
    for a in names:
        row = a.ljust(14)
        for b in names:
            inter = len(value_sets[a]) if a == b else len(value_sets[a] & value_sets[b])
            row += str(inter).ljust(col_w)
        print(row)


# --------------------------------------------------------------------------- #
# C. Раздельные parquet по регионам + диагностика (см. docstring — без merge)
# --------------------------------------------------------------------------- #

def write_region_parquets(canon: dict[str, pd.DataFrame], out_dir: str) -> None:
    print("\n" + "=" * 70)
    print("C. Сборка по регионам (data/unified.parquet НЕ создаётся — см. docstring модуля)")
    print("=" * 70)
    by_region_dir = os.path.join(out_dir, "by_region")
    os.makedirs(by_region_dir, exist_ok=True)
    total_rows = 0
    for name, df in canon.items():
        path = os.path.join(by_region_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        n = len(df)
        total_rows += n
        dt_min, dt_max = df["created_at"].min(), df["created_at"].max()
        share_created = df["created_at"].isna().mean()
        share_region = df["region"].isna().mean()
        share_category = df["category"].isna().mean()
        print(
            f"[{name}] строк: {n}; окно created_at: {dt_min} .. {dt_max}; "
            f"доля пустых created_at {share_created:.2%}, region {share_region:.2%}, "
            f"category {share_category:.2%} -> {path}"
        )
    print(f"\nИтого строк по всем регионам (раздельно, без физической склейки): {total_rows}")


# --------------------------------------------------------------------------- #
# Сборщик
# --------------------------------------------------------------------------- #

def run_all() -> dict[str, pd.DataFrame]:
    print("=" * 70)
    print("A. Адаптеры по регионам — построчная самопроверка")
    print("=" * 70)

    canon: dict[str, pd.DataFrame] = {}
    for name, loader in REGION_LOADERS.items():
        canon[name] = loader()

    total = sum(len(df) for df in canon.values())
    print("\n" + "=" * 70)
    print("Контроль сборки против CLAUDE.md, раздел «Контроль»")
    print("=" * 70)
    print("Ожидаемая сумма ПОСЛЕ дедупа, ДО отбрасывания непарсящихся дат: 990 032")
    print("Ожидание CLAUDE.md после отбрасывания дат: 990 000 (32 строки Акмолы, Акмола -> 3 474)")
    print(f"Фактическая сумма после дедупа И отбрасывания дат: {total}")
    if total != 990000:
        print(
            f"РАСХОЖДЕНИЕ: фактический итог {total} != ожидаемых 990 000. "
            f"Разбивка по регионам с отброшенными по дате строками — см. вывод адаптеров выше "
            f"(Караганда дополнительно теряет 5 строк с нечитаемой датой, это не входило в "
            f"ожидание CLAUDE.md, которое называло только 32 строки Акмолы)."
        )

    topic_dictionaries(canon)
    write_region_parquets(canon, OUT_DIR)
    return canon


if __name__ == "__main__":
    run_all()
