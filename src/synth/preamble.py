"""Обязательная первая строка любого отчёта по синтетическому корпусу (5n)."""
from pathlib import Path

PREAMBLE = (
    "Это проверка конвейера на синтетических обращениях, а не оценка качества. "
    "Тексты написал генератор, метки поставлены по тому же правилу, которое "
    "задавало текст. Высокие числа значат только одно: модель воспроизводит "
    "генератор. О качестве на настоящих обращениях они не говорят ничего. "
    "Казахская часть носителем языка не проверялась."
)


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
