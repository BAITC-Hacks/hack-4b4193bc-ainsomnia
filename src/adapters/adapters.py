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

Почему сводную таблицу собирает отдельный модуль
------------------------------------------------
Этот модуль пишет по одному parquet на регион (`data/by_region/<регион>.parquet`)
и сводную таблицу НЕ создаёт. Её собирает следующий шаг —
`src/adapters/build_unified.py`; порядок запуска описан в CLAUDE.md,
раздел «Развёртывание с нуля».

Разделение сохранено намеренно, но причина у него теперь другая, чем при
написании модуля. Тогда сборка `data/unified.parquet` считалась запрещённой
заморозкой из раздела 3 CLAUDE.md. **Решение принято и записано** (раздел 3,
«Уточнение области запрета»): запрет касается обучения общей МОДЕЛИ, а не
сводных таблиц, поэтому сводная таблица разрешена и собрана.

Что осталось от прежней причины: объединять регионы в одну обучающую выборку
по-прежнему нельзя до объяснения расхождения 34,9% / 7,5% (раздел 3). Отдельный
модуль сборки удерживает эту границу видимой — порегиональные parquet остаются
первичными, а сводная таблица является витринной, не обучающей.
"""

from __future__ import annotations

import os
import re

import pandas as pd

from src import paths
from src.checks.field_shift import (almaty_shift_rules, karaganda_shift_rules,
                                     vko_shift_rules)


# Каталог выгрузки и результатов — src/paths.py (CLAUDE.md, 5p): настоящая выгрузка
# по умолчанию, поддельная — при NAZAR_SOURCE=fake.
BASE_DIR = str(paths.RAW_DIR)
OUT_DIR = str(paths.DATA_DIR)

# Ожидаемый итог сводной таблицы (CLAUDE.md, раздел 5c; для поддельной выгрузки — 5p).
# Меняется вместе с правилами отсева — при правке обновлять здесь и в CLAUDE.md одновременно.
EXPECTED_TOTAL = 12_910 if paths.FAKE else 988_776

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
if paths.FAKE:   # поддельная выгрузка, CLAUDE.md, 5p
    INVENTORY_RAW_ROWS = {"Караганда": 10000, "Костанай": 500, "Туркестан": 450, "ВКО": 500,
                          "Алматы": 500, "Акмола": 400, "Павлодар": 600}


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #

def _read_csv(path: str) -> pd.DataFrame:
    # все загрузчики читают сырьё здесь — одна точка для понятного сообщения
    from src.cli import require_raw
    require_raw(path)
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
    df_valid, parsed_valid, shift_dropped = _drop_shifted(df_valid, parsed_valid, name, karaganda_shift_rules)

    src = pd.Series(os.path.basename(path), index=df_valid.index)
    canon, absent = _build_canon(
        df_valid, parsed_valid, "Карагандинская область",
        category_col="sub_category", district_col="district",
        executor_col="executor_gov_org", status_col=None, sla_breach_col=None,
        source_file=src, name=name,
    )
    return _finish(name, n_raw, dedup_dropped, date_dropped, canon, absent,
                   shift_dropped=shift_dropped)


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
    df_valid, parsed_valid, shift_dropped = _drop_shifted(df_valid, parsed_valid, name, vko_shift_rules)

    src = pd.Series(os.path.basename(path), index=df_valid.index)
    canon, absent = _build_canon(
        df_valid, parsed_valid, "Восточно-Казахстанская область",
        category_col="category", district_col="district",
        executor_col="contractor", status_col="status", sla_breach_col=None,
        source_file=src, name=name,
    )
    return _finish(name, n_raw, dedup_dropped, date_dropped, canon, absent,
                   shift_dropped=shift_dropped)


# Сдвиг полей на незакрытых кавычках (CLAUDE.md, 5c): содержимое одной колонки
# уезжает в другую. Один дефект — одно обращение: во всех регионах, где он найден,
# строки со сдвигом отсеиваются в адаптере. Правила — в src/checks/field_shift.py,
# там же отчёт, который воспроизводит эти счётчики по сырым файлам.
def _drop_shifted(df: pd.DataFrame, parsed: pd.Series, name: str, rules_fn):
    rules = rules_fn(df)
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
    df_valid, parsed_valid, shift_dropped = _drop_shifted(df_valid, parsed_valid, name, almaty_shift_rules)

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
    paths.validate_raw_source(BASE_DIR, recursive=True)
    paths.validate_work_dir(paths.WORK)
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
    if paths.FAKE:
        print(f"ПОДДЕЛЬНАЯ ВЫГРУЗКА. Ожидание CLAUDE.md (раздел 5p): {EXPECTED_TOTAL} = 12 950 - 3 "
              "(Акмола, непарсящаяся дата) - 2 (Караганда, непарсящаяся дата) - 20 (Караганда, "
              "сдвиг полей) - 12 (Алматы, сдвиг полей) - 3 (ВКО, сдвиг полей)")
    else:
        print("Ожидаемая сумма ПОСЛЕ дедупа, ДО прочих отсевов: 990 032")
        print(f"Ожидание CLAUDE.md (раздел 5c): {EXPECTED_TOTAL} = 990 032 - 32 (Акмола, "
              "непарсящаяся дата) - 5 (Караганда, непарсящаяся дата) - 638 (Караганда, сдвиг "
              "полей) - 578 (Алматы, сдвиг полей) - 3 (ВКО, сдвиг полей)")
    print(f"Фактическая сумма: {total}")
    if total != EXPECTED_TOTAL:
        print(
            f"РАСХОЖДЕНИЕ: фактический итог {total} != ожидаемых {EXPECTED_TOTAL}. "
            f"Разбивка по регионам — см. вывод адаптеров выше: у каждого напечатано, "
            f"сколько снято дедупом, отброшено по дате и отсеяно по сдвигу полей."
        )

    topic_dictionaries(canon)
    write_region_parquets(canon, OUT_DIR)
    paths.write_source_marker(paths.SOURCE_MARK, BASE_DIR)
    return canon


if __name__ == "__main__":
    run_all()
