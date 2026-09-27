"""Descriptive comparison of successful dataset versions; thresholds: CLAUDE 0e."""
ROW_CHANGE = .20
SHARE_CHANGE = .10
WINDOW_SHRINK = .20


def compare(current, previous):
    if previous is None:
        return {"available": False, "reason": "Нет предыдущей успешной версии", "changes": []}
    if current["source"] != previous["source"]:
        raise ValueError("Cannot compare different sources")
    if current.get("mapping_revision") != previous.get("mapping_revision"):
        return {"available": False, "reason": "Изменилась версия разметки тем; сравнение входного потока недоступно",
                "previous_dataset_id": previous["dataset_id"], "changes": []}
    changes = []

    def add(region, field, before, after, warning):
        if before != after:
            changes.append({"region": region, "field": field, "before": before,
                            "after": after, "warning": bool(warning)})

    old, new = previous["regions"], current["regions"]
    for region in sorted(old.keys() | new.keys()):
        if region not in old or region not in new:
            add(region, "регион в выгрузке", region in old, region in new, True)
            continue
        a, b = old[region], new[region]
        n0, n1 = a["rows"], b["rows"]
        add(region, "число строк", n0, n1, abs(n1 - n0) >= ROW_CHANGE * n0)
        for key in ("class_shares", "topic_shares", "completeness"):
            for field in sorted(a[key].keys() | b[key].keys()):
                x, y = a[key].get(field, 0), b[key].get(field, 0)
                add(region, key + ":" + field, x, y, abs(y - x) + 1e-12 >= SHARE_CHANGE)
        ca, cb = a["categories"], b["categories"]
        added, removed = cb.keys() - ca.keys(), ca.keys() - cb.keys()
        if added or removed:
            changes.append({"region": region, "field": "состав категорий",
                            "before": {"removed": len(removed)}, "after": {"added": len(added)},
                            "warning": True})
        # Changes to frequencies are also visible, even when category names stay the same.
        changed = sum(ca.get(key, 0) != cb.get(key, 0) for key in ca.keys() | cb.keys())
        for key in sorted(ca.keys() | cb.keys()):
            add(region, "category_sha256:" + key, ca.get(key, 0), cb.get(key, 0), False)
        if changed:
            changes.append({"region": region, "field": "категорий с изменившейся частотой",
                            "before": 0, "after": changed, "warning": False})
        for field in ("first_date", "last_date", "history_days"):
            warn = field == "history_days" and b[field] <= a[field] * (1 - WINDOW_SHRINK)
            add(region, field, a[field], b[field], warn)
    return {"available": True, "previous_dataset_id": previous["dataset_id"], "changes": changes}
