"""Aggregate health profile produced only after a successful labeled build (CLAUDE 0e)."""
from datetime import datetime, timezone
import hashlib
import json
import math

import pandas as pd

from src import paths, forecast
from src.data_drift import compare
from src.dataset_manifest import digest, generate, verify, write_json
from src.checks.labeling import BASE, EMPTY
from src.topic_mapping import mapping_revision

PROFILE = paths.DATA_DIR / "data_health.json"
PREVIOUS = paths.DATA_DIR / "data_health_previous.json"
BUILD_STATE = paths.DATA_DIR / "last_build.json"
FIELDS = ("created_at", "region", "category", "topic", "appeal_class", "district",
          "executor", "status", "sla_breach")
REGIONS = {"Акмолинская область", "Алматинская область", "Восточно-Казахстанская область",
           "Карагандинская область", "Костанайская область", "Павлодарская область", "Туркестанская область"}


def validate_profile(value):
    """Reject malformed persisted profiles before passing them to the UI or drift."""
    def require(ok):
        if not ok:
            raise ValueError("Invalid aggregate profile")
    def sha(value):
        return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
    require(value["format_version"] == 1 and value["source"] in ("real", "fake"))
    require(sha(value["dataset_id"]) and sha(value["unified_sha256"]))
    if "mapping_revision" in value:
        require(sha(value["mapping_revision"]))
    require(value["schema_status"] == "compatible" and value["fingerprint_status"] == "verified_at_build")
    datetime.fromisoformat(value["built_at"])
    require(bool(value["regions"]) and set(value["regions"]) <= REGIONS)
    for item in value["regions"].values():
        require(type(item["rows"]) is int and item["rows"] > 0)
        first, last = [datetime.fromisoformat(item[k]) for k in ("first_date", "last_date")]
        require(item["history_days"] == (last - first).days + 1)
        require(0 < item["observed_days"] <= item["history_days"])
        require(0 <= item["longest_gap"] < item["history_days"])
        require(set(item["completeness"]) == set(FIELDS))
        require(set(item["class_shares"]) == {"problem", "info", "system"})
        require(abs(sum(item["class_shares"].values()) - 1) < 1e-9)
        for key in ("completeness", "class_shares", "topic_shares"):
            require(all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1
                        for v in item[key].values()))
        require(all(sha(k) and type(v) is int and v > 0 for k, v in item["categories"].items()))
        require(sum(item["categories"].values()) == item["rows"])
        require(type(item["new_categories"]) is int and 0 <= item["new_categories"] <= len(item["categories"]))
        require(item["status"] in ("OK", "WARNING", "BLOCKED"))
        require(isinstance(item["reasons"], list) and all(isinstance(v, str) for v in item["reasons"]))
        require(type(item["forecast"]["available"]) is bool and isinstance(item["forecast"]["reason"], str))
        for gap in item["gaps_ge7"]:
            a, b = [datetime.fromisoformat(gap[k]) for k in ("first_date", "last_date")]
            require(first < a <= b < last and gap["days"] == (b - a).days + 1 and gap["days"] >= 7)
        if item["discarded"] is not None:
            require(all(type(item["discarded"][k]) is int and item["discarded"][k] >= 0 for k in
                        ("raw_rows", "dedup_dropped", "date_dropped", "field_shift_dropped", "output_rows", "date_shift_proxy")))
    require(type(value["drift"]["available"]) is bool)
    if not value["drift"]["available"]:
        require(value["drift"]["reason"] in ("Нет предыдущей успешной версии",
                "Изменилась версия разметки тем; сравнение входного потока недоступно"))
    for row in value["drift"]["changes"]:
        require(row["region"] in REGIONS and type(row["warning"]) is bool)
        for key in ("field", "before", "after"):
            require(key in row)


def missing_runs(dates):
    days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    if not len(days):
        return []
    runs = []
    for before, after in zip(days[:-1], days[1:]):
        length = (after - before).days - 1
        if length:
            runs.append({"first_date": (before + pd.Timedelta(days=1)).date().isoformat(),
                         "last_date": (after - pd.Timedelta(days=1)).date().isoformat(), "days": length})
    return runs


def forecast_status(problem, region, source):
    series = forecast.weekly(problem, region)
    weeks = len(series)
    gap = forecast.longest_gap(series)[0]
    minimum = forecast.SEASON_LAG + 13 + forecast.LEVEL_WINDOW
    if source == "fake":
        available, reason = False, "FAKE: рабочий прогноз не применяется"
    elif gap > forecast.MAX_GAP:
        available, reason = False, f"Пропуск {gap} недель; допустимо не более {forecast.MAX_GAP}"
    elif weeks < minimum:
        available, reason = False, f"История {weeks} недель; нужно минимум {minimum}"
    elif region not in forecast.REGIONS:
        available, reason = False, "Истории хватает, но регион не включён в существующий forecast"
    else:
        available = True
        reason = f"История {weeks} недель; отсечек проверки: {min(3, forecast.max_folds(weeks, 13))}"
    if region == "Карагандинская область" and len(series) and series.index.max().year <= 2023:
        reason += "; историческая проверка метода, не текущий прогноз"
    return {"available": available, "reason": reason, "weeks": weeks}


def profile(frame, manifest, counts, *, source, built_at):
    base = json.loads(BASE.read_text())
    allowed_topics = {value[0] for value in base.values()}
    if (not set(FIELDS) <= set(frame) or frame.empty
            or not set(frame.region.dropna()) <= REGIONS
            or not set(frame.topic.dropna()) <= allowed_topics
            or not set(frame.appeal_class.dropna()) <= {"problem", "info", "system"}):
        raise ValueError("Canonical schema incompatible")
    if frame[list(FIELDS[:2]) + ["topic", "appeal_class"]].isna().any().any():
        raise ValueError("Required fields incomplete")
    problem = frame[frame.appeal_class == "problem"]
    output = {"format_version": 1, "source": source, "dataset_id": manifest["dataset_id"],
              "mapping_revision": mapping_revision(),
              "built_at": built_at, "schema_status": "compatible",
              "fingerprint_status": "verified_at_build", "regions": {}}
    for region, group in frame.groupby("region"):
        first, last = group.created_at.min().normalize(), group.created_at.max().normalize()
        runs = missing_runs(group.created_at)
        categories = group.category.fillna(EMPTY).value_counts()
        p = group[group.appeal_class == "problem"]
        completeness = {field: float((group[field].notna() & group[field].astype(str).str.strip().ne("")).mean())
                        for field in FIELDS}
        item = {"rows": len(group), "first_date": first.date().isoformat(),
                "last_date": last.date().isoformat(), "history_days": (last - first).days + 1,
                "observed_days": int(group.created_at.dt.normalize().nunique()),
                "longest_gap": max((r["days"] for r in runs), default=0),
                "gaps_ge7": [r for r in runs if r["days"] >= 7],
                "completeness": completeness,
                "class_shares": {key: float((group.appeal_class == key).mean()) for key in ("problem", "info", "system")},
                "topic_shares": {key: float(value / len(p)) for key, value in p.topic.value_counts().items()},
                "categories": {hashlib.sha256(str(key).encode()).hexdigest(): int(value)
                               for key, value in categories.items()},
                "new_categories": sum(key not in base for key in categories.index),
                "discarded": counts.get(region), "forecast": forecast_status(problem, region, source),
                "status": "OK", "reasons": []}
        if item["gaps_ge7"]:
            item["reasons"].append(f"Пропуски ≥7 дней: {len(item['gaps_ge7'])}; наибольший {item['longest_gap']} дней")
        if item["new_categories"]:
            item["reasons"].append(f"Новых категорий: {item['new_categories']}")
        if completeness["category"] < .95:
            item["reasons"].append("Категория заполнена менее чем у 95% строк")
        discarded = item["discarded"]
        if discarded is None:
            item["reasons"].append("Нет счётчиков отсева этой сборки")
        else:
            calculated = discarded["raw_rows"] - sum(discarded[k] for k in ("dedup_dropped", "date_dropped", "field_shift_dropped"))
            if discarded["output_rows"] != len(group) or calculated != len(group):
                raise ValueError("Adapter counts inconsistent")
            if discarded["field_shift_dropped"] or discarded["date_dropped"]:
                item["reasons"].append(f"Отсев сдвига полей: {discarded['field_shift_dropped']}; по дате: {discarded['date_dropped']}")
        if source == "real" and len(p):
            top = p.topic.value_counts().index[0]
            years = p.created_at.dt.year
            share_first = float((p.loc[years == years.min(), "topic"] == top).mean())
            share_last = float((p.loc[years == years.max(), "topic"] == top).mean())
            if abs(share_last - share_first) * 100 > forecast.MIX_SHIFT:
                item["reasons"].append(f"Состав потока: доля главной темы между крайними годами изменилась более чем на {forecast.MIX_SHIFT} п.п.; причину нужно уточнить")
                item["forecast"]["reason"] += "; состав тем менялся — разрез по темам требует проверки"
        if item["reasons"]:
            item["status"] = "WARNING"
        output["regions"][region] = item
    return output


def build():
    paths.require_writable()
    paths.require_source_marker(paths.SOURCE_MARK)
    manifest = (generate(paths.RAW_DIR, include_xlsx=False, source="fake") if paths.FAKE else verify())
    state = json.loads(BUILD_STATE.read_text())
    if (state["state"] != "running" or state["stage"] != 5 or state["source"] != paths.SOURCE
            or state["dataset_id"] != manifest["dataset_id"]):
        raise ValueError("Health must follow the verified build of this input version")
    frame = pd.read_parquet(paths.UNIFIED, columns=list(FIELDS))
    frame.created_at = pd.to_datetime(frame.created_at)
    counts = json.loads((paths.DATA_DIR / "build_counts.json").read_text())
    current = profile(frame, manifest, counts, source=paths.SOURCE,
                      built_at=datetime.now(timezone.utc).isoformat())
    current["unified_sha256"] = digest(paths.UNIFIED)
    previous = None
    if PROFILE.exists():
        old = json.loads(PROFILE.read_text())
        validate_profile(old)
        if old["source"] != current["source"]:
            raise ValueError("Profile source mismatch")
        if (old["dataset_id"] != current["dataset_id"]
                or old.get("mapping_revision") != current["mapping_revision"]):
            previous = old
        elif PREVIOUS.exists():
            previous = json.loads(PREVIOUS.read_text())
    if previous is not None:
        validate_profile(previous)
    current["drift"] = compare(current, previous)
    for region, item in current["regions"].items():
        warnings = [c for c in current["drift"]["changes"] if c["region"] == region and c["warning"]]
        if warnings:
            item["status"] = "WARNING"
            item["reasons"].append(f"Изменений выше порогов drift: {len(warnings)}; см. сравнение версий")
    validate_profile(current)
    if previous is not None:
        write_json(PREVIOUS, previous)
    write_json(PROFILE, current)
    print(f"Data Health: {len(current['regions'])} регионов; строк {len(frame)}; source={paths.SOURCE}")
    return current


def load():
    """Return (safe profile, blocking reason); never expose an exception payload."""
    paths.require_source_marker(paths.SOURCE_MARK)
    try:
        if BUILD_STATE.exists():
            state = json.loads(BUILD_STATE.read_text())
            if state["source"] != paths.SOURCE or state["state"] != "ok":
                return None, "Последняя сборка не завершена успешно; требуется проверка инженера"
        if not PROFILE.exists():
            return None, None
        current = json.loads(PROFILE.read_text())
        validate_profile(current)
        if current["source"] != paths.SOURCE:
            return None, "Источник профиля не совпадает с SOURCE"
        if current.get("mapping_revision") != mapping_revision():
            return None, "Изменилась версия разметки тем; повторите сборку"
        if current["unified_sha256"] != digest(paths.UNIFIED):
            return None, "Fingerprint канона не совпадает с профилем; повторите сборку"
        if not paths.FAKE:
            accepted = json.loads((paths.DATA_DIR / "raw_manifest.json").read_text())
            if accepted["dataset_id"] != current["dataset_id"]:
                return None, "Принята другая версия сырья, её успешной сборки ещё нет"
        if set(current["regions"]) - REGIONS:
            return None, "Несовместимая схема профиля"
        return current, None
    except (OSError, ValueError, KeyError, TypeError):
        return None, "Профиль качества повреждён или несовместим; повторите сборку"


if __name__ == "__main__":
    try:
        build()
    except Exception:
        print("Не удалось построить профиль качества; схема или артефакты несовместимы.")
        raise SystemExit(2) from None
