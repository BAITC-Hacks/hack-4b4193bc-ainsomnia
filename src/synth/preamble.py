"""Обязательная первая строка любого отчёта по синтетическому корпусу (5n)."""
from pathlib import Path

PREAMBLE = (
    "Это проверка конвейера на синтетических обращениях, а не оценка качества. "
    "Тексты написал генератор, метки поставлены по тому же правилу, которое "
    "задавало текст. Высокие числа значат только одно: модель воспроизводит "
    "генератор. О качестве на настоящих обращениях они не говорят ничего. "
    "Казахская часть носителем языка не проверялась."
)


def is_template_set(ids) -> bool:
    """Набор целиком из шаблонного генератора C (id вида C-…)."""
    ids = [str(i) for i in ids]
    return bool(ids) and all(i.startswith("C-") for i in ids)


def c_caveat(effective: int, n: int) -> str:
    """Оговорка к любому числу, посчитанному на C: шаблоны повторяют одну
    ситуацию с другим городом, и различных текстов меньше, чем строк."""
    return (f"посчитано на шаблонах C: после схлопывания почти-дублей "
            f"(Jaccard 5-грамм ≥ 0.8) различных текстов {effective} из {n} — "
            f"эффективный объём теста меньше {n}")


def with_preamble(text: str, synthetic: bool) -> str:
    if not synthetic:
        return text
    report = PREAMBLE + "\n\n" + text
    if not report.startswith(PREAMBLE):
        raise AssertionError("отчёт по синтетике обязан начинаться преамбулой")
    return report


def require_flag(csv: str, synthetic: bool) -> None:
    path = Path(csv).resolve()
    if path.is_relative_to(Path("data/synth").resolve()) and not synthetic:
        raise ValueError("для корпуса data/synth/ нужен --synthetic: "
                         "отчёт обязан начинаться преамбулой")
