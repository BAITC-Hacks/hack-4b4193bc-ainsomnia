"""Агрегатный отчёт о синтетическом корпусе; первая строка обязательна."""
from __future__ import annotations

import json
from pathlib import Path

from src.synth.preamble import PREAMBLE

MANIFEST = Path("tests/fixtures/synth/manifest.json")
OUT = Path("reports/synth/corpus.md")


def render(manifest: dict) -> str:
    if manifest.get("scope") != PREAMBLE:
        raise ValueError("опись синтетики не содержит обязательную преамбулу")
    split = manifest["split"]
    c_eff = manifest["c_near_duplicates"]
    c_lang = c_eff["effective_by_language"]
    privacy = manifest["privacy"]
    review = privacy["manual_review"]
    if not isinstance(review, dict) or review.get("a_read") != 200 or review.get("c_read") != 100:
        raise ValueError("не записана ручная проверка 200 A и 100 C")
    lines = [
        PREAMBLE,
        "",
        "# Синтетический корпус 109",
        "",
        f"A: {manifest['a_rows']:,} текстов; C: {manifest['c_rows']:,}; "
        f"классов: {len(manifest['classes'])}.",
        "",
        "| Часть A | Текстов | Происшествий |",
        "|---|---:|---:|",
    ]
    for part in ("train", "validation", "test", "seed_holdout"):
        lines.append(f"| {part} | {split['split_counts'][part]} | "
                     f"{split['incident_counts'][part]} |")
    lines += [
        "",
        f"Найдено почти-дублей внутри A: {split['near_duplicate_pairs']} пар; "
        "они объединяются в группы до разбиения. "
        f"Между A и C: {manifest['a_c_near_duplicate_groups']} групп.",
        "",
        f"**Эффективный объём C — {c_eff['effective_texts']} различных текстов "
        f"из {manifest['c_rows']:,}** (русских {c_lang['ru']}, казахских "
        f"{c_lang['kk']} — из {manifest['c_rows'] // 2} каждого языка): внутри C "
        f"{c_eff['near_duplicate_pairs']:,} пар почти-дублей по тому же порогу "
        "Jaccard 5-грамм ≥ 0.8, после схлопывания остаётся столько групп. "
        "Причина — построение шаблонов, не сбой подстановки: на класс и язык "
        "3 ситуации × 4 рамки, в трёх рамках из четырёх меняется только город, "
        "и ни одна группа не смешивает разные ситуации или рамки. По содержанию "
        "C — 3 ситуации на класс и язык. Любое число, посчитанное на C, "
        "относится к этому эффективному объёму, а не к 1 200.",
        "",
        "| Шум | Применено | Подходящих текстов | Доля |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("Опечатки", "typo"), ("Без пунктуации", "no_punct"),
                       ("Замена казахских букв", "kk_letters")):
        item = manifest["noise"][key]
        lines.append(f"| {label} | {item['count']} | {item['denominator']} | "
                     f"{item['count'] / item['denominator']:.1%} |")
    lines += [
        "",
        f"Автоматических срабатываний ПДн: {privacy['automated_hits']}; "
        f"общих цепочек из 8 слов с реальным свободным текстом: "
        f"{privacy['real_text_overlap_8_words']} "
        f"(проверено {privacy['real_values_scanned']:,} непустых значений).",
        f"Ручное чтение: {review['a_read']} A и {review['c_read']} C; "
        f"подтверждённых ПДн {review['confirmed_pii']}; "
        f"явных противоречий метке {review['clear_label_conflicts']}. "
        f"Пограничные случаи: {', '.join(review['borderline_labels'])}.",
        "",
        "Число текстов, удалённых при раннем написании A, не записывалось. "
        "Казахские тексты носителем языка не проверялись. "
        "Шаблоны C повторяются по построению.",
        "",
        "Модели на этом корпусе ещё не обучались; чисел качества здесь нет.",
        "",
    ]
    if lines[0] != PREAMBLE:
        raise AssertionError("отчёт должен начинаться преамбулой")
    return "\n".join(lines)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = render(manifest)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
