"""Translate user-entered shifts into the existing 13-week capacity contract."""
import math
from numbers import Real

from src.planning import capacity

HORIZON_WEEKS = 13
MAX_RESERVE_PCT = 95.0
RESERVE_UTILIZATION = 0.85
FULL_UTILIZATION = 1.0


def shift_capacity(demand, operators_per_shift, hours_per_shift, shifts_per_week,
                   avg_handling_minutes, reserve_pct, *, forecast_weeks):
    values = (demand, operators_per_shift, hours_per_shift, shifts_per_week,
              avg_handling_minutes, reserve_pct, forecast_weeks)
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
        raise ValueError('Заполните все параметры конечными числами.')
    if forecast_weeks != HORIZON_WEEKS:
        raise ValueError('Для сравнения нужен прогноз за те же 13 недель.')
    if demand < 0:
        raise ValueError('Ожидаемый поток не может быть отрицательным.')
    if operators_per_shift <= 0 or int(operators_per_shift) != operators_per_shift:
        raise ValueError('Укажите целое число операторов в смене больше нуля.')
    if not 0 < hours_per_shift <= 24:
        raise ValueError('Часов в смене должно быть больше нуля и не больше 24.')
    if shifts_per_week <= 0:
        raise ValueError('Число смен в неделю должно быть больше нуля.')
    if avg_handling_minutes <= 0:
        raise ValueError('Среднее время обработки должно быть больше нуля минут.')
    if not 0 <= reserve_pct <= MAX_RESERVE_PCT:
        raise ValueError('Резерв мощности должен быть от 0 до 95%.')
    hours = hours_per_shift * shifts_per_week * HORIZON_WEEKS
    if not math.isfinite(hours):
        raise ValueError('Параметры слишком велики для расчёта; уменьшите значения.')
    rows = capacity(demand, operators_per_shift, hours, avg_handling_minutes,
                    reserve_pct / 100, weeks=forecast_weeks)
    for row in rows:
        if row['available_capacity'] <= 0 or any(
            row[k] is None or not math.isfinite(row[k])
            for k in ('demand', 'raw_capacity', 'available_capacity', 'utilization',
                      'balance', 'required_operator_hours')
        ):
            raise ValueError('Параметры выходят за числовой диапазон; проверьте значения.')
    return rows


def utilization_status(utilization):
    """Display thresholds only; never change scenario values or forecast."""
    if utilization <= RESERVE_UTILIZATION:
        return 'Есть резерв', 'ready'
    if utilization <= FULL_UTILIZATION:
        return 'Высокая загрузка', 'attention'
    return 'Мощности недостаточно', 'attention'
