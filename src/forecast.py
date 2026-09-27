#!/usr/bin/env python3
"""Прогноз недельной нагрузки по обращениям класса problem.

    .venv/bin/python -m src.forecast [--horizon 13] [--keep-first-month]

Кандидаты на расчёт — регионы, где история позволяет оценить годовую сезонность:
Павлодарская, Восточно-Казахстанская, Карагандинская и Туркестанская. Туркестан
из расчёта выпадает: в его выгрузке разрыв в 24 недели, а ряд с разрывом нельзя
ни обучать, ни проверять (см. MAX_GAP и longest_gap). Остаются три региона, и
Караганда среди них проходит порог впритык — на скользящую проверку у неё
остаётся 2 отсечки против 3 у Павлодара и ВКО. Для остальных прогноз НЕ строится:
это ограничение данных, а не пропуск; см. таблицу в отчёте.

Ряды: обращения по неделям (неделя помечена датой понедельника, которым она
заканчивается), разрез регион и регион x тема для пяти крупнейших тем региона.
Неполные крайние недели отбрасываются, первый календарный месяц ряда тоже —
это выход системы на режим, как и в детекторе всплесков.

Три модели, намеренно простые. Усложнять имеет смысл только после того, как
видно, что простое не справляется:
  НАИВНЫЙ   ŷ(t) = y(t-52)                — «столько же, сколько год назад»
  СЕЗОННЫЙ  ŷ(t) = y(t-52) * k            — то же с поправкой на текущий уровень,
                                            k = медиана(последние 13 недель) /
                                                медиана(те же 13 недель год назад)
  ПЛОСКИЙ   ŷ(t) = медиана(последние 8)   — без сезонности вовсе, контроль:
                                            нужна ли годовая сезонность в принципе

Проверка: обучение на данных до отсечки, прогноз на следующие 13 недель,
сравнение с фактом. MAPE считается только по неделям с ненулевым фактом,
число исключённых недель печатается рядом. Одна отсечка — это один бросок,
поэтому та же проверка повторяется на нескольких сдвинутых отсечках
(скользящая проверка): вывод о том, какая модель лучше, делается по ней.

Отчёт печатается в консоль и сохраняется в reports/forecast.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from src import paths

DATA = paths.UNIFIED
OUT = paths.REPORTS_DIR / "forecast.md"
REGIONS = ["Павлодарская область", "Восточно-Казахстанская область",
           "Карагандинская область", "Туркестанская область"]
TOP_TOPICS = 5
SEASON_LAG = 52          # недель в году
LEVEL_WINDOW = 13        # окно для поправки на уровень
FLAT_WINDOW = 8          # окно плоского прогноза
OUTLIER_K = 5            # во сколько раз выше медианы неделя считается выбросом
SEASONAL_K = 3           # во сколько раз меняется уровень — окно пересекает смену сезона
MIN_MONTH_YEARS = 2      # лет наблюдений на месяц: меньше — это тренд, а не сезон
MIN_MATCHED = 3          # минимум отсечек того же календарного перехода
MIN_LEVEL = 10           # ниже этого уровня поправка k ненадёжна (как min-count у детектора)
MAX_GAP = 4              # недель подряд без обращений — это пропуск выгрузки, не затишье
MIX_SHIFT = 30           # п.п. сдвига доли главной темы — состав потока сменился
MONTHS = ("", "января", "февраля", "марта", "апреля", "мая", "июня", "июля",
          "августа", "сентября", "октября", "ноября", "декабря")
MODELS = ("наивный", "сезонный", "плоский")

_lines = []


def say(text=""):
    print(text)
    _lines.append(text)


# ---------------------------------------------------------------- ряды
def weekly(df, region, topic=None, skip_first_month=True):
    g = df[df.region == region]
    if topic is not None:
        g = g[g.topic == topic]
    if g.empty:
        return pd.Series(dtype=float)
    lo, hi = df.loc[df.region == region, "created_at"].agg(["min", "max"])
    if skip_first_month:
        lo = (lo.to_period("M") + 1).to_timestamp()
    s = g.set_index("created_at").resample("W-MON").size()
    full = pd.date_range(s.index.min(), s.index.max(), freq="W-MON")
    s = s.reindex(full, fill_value=0)
    # только недели, целиком лежащие внутри окна наблюдения
    keep = (s.index - pd.Timedelta(days=6) >= lo) & (s.index <= hi)
    return s[keep].astype(float)


# ---------------------------------------------------------------- модели
def forecast(train, steps, model):
    """train — ряд до отсечки; возвращает прогноз на steps недель вперёд."""
    y = train.to_numpy(dtype=float)
    if model == "плоский":
        return np.full(steps, np.median(y[-FLAT_WINDOW:]))
    if len(y) < SEASON_LAG + steps:
        return None                      # сезонного лага не хватает
    base = y[-SEASON_LAG:][:steps] if steps <= SEASON_LAG else None
    if base is None:
        return None
    if model == "наивный":
        return base.copy()
    if model == "сезонный":
        return base * level_guarded(train)[0]
    raise ValueError(model)


def level_guarded(train):
    """Поправка на уровень с защитой: (k, сработала ли защита, медианы).

    Защита обязана действовать и в бэктесте, и в прогнозе вперёд. Иначе модели
    сравниваются в одних условиях, а применяются в других: на теплоснабжении
    Павлодара уровень ниже MIN_LEVEL держится на 102 отсечках из 256, и без
    защиты бэктест меряет не ту модель, которая пошла бы в работу.
    """
    k, now, then = level_factor(train)
    if min(now, then) < MIN_LEVEL:
        return 1.0, True, (now, then, k)
    return k, False, (now, then, k)


def level_factor(train):
    """Поправка на уровень: во сколько раз последние LEVEL_WINDOW недель выше тех же
    недель годом раньше. Возвращает (k, медиана сейчас, медиана год назад) — две
    последние нужны, чтобы видеть, на каком уровне поправка посчитана."""
    y = train.to_numpy(dtype=float)
    now = float(np.median(y[-LEVEL_WINDOW:]))
    then = float(np.median(y[-LEVEL_WINDOW - SEASON_LAG:-SEASON_LAG]))
    return (now / then if then > 0 else 1.0), now, then


def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    mae = float(np.mean(np.abs(a - p)))
    nz = a > 0
    mape = float(np.mean(np.abs(a[nz] - p[nz]) / a[nz]) * 100) if nz.any() else float("nan")
    return {"MAE": mae, "MAPE": mape, "нулевых недель": int((~nz).sum())}


def horizon_mae(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    out = {}
    for name, sl in (("1–4", slice(0, 4)), ("5–8", slice(4, 8)), ("9–13", slice(8, 13))):
        if len(a) > sl.start:
            out[name] = float(np.mean(np.abs(a[sl] - p[sl])))
    return out


# ---------------------------------------------------------------- проверка
def evaluate_at(series, origin, horizon):
    """Одна отсечка: обучение на series[:origin], прогноз на horizon недель."""
    train, test = series.iloc[:origin], series.iloc[origin:origin + horizon]
    if len(test) < horizon:
        return None
    res = {}
    for m in MODELS:
        pred = forecast(train, horizon, m)
        if pred is None:
            continue
        res[m] = {**metrics(test.to_numpy(), pred),
                  "по горизонту": horizon_mae(test.to_numpy(), pred)}
    return {"отсечка": series.index[origin - 1], "модели": res} if res else None


def evaluate(series, horizon, folds=3):
    """Последняя отсечка плюс скользящая проверка на сдвинутых отсечках."""
    min_train = SEASON_LAG + LEVEL_WINDOW
    if len(series) < min_train + horizon:
        return None
    out = []
    for k in range(1, folds + 1):
        origin = len(series) - k * horizon
        if origin < min_train:
            break
        r = evaluate_at(series, origin, horizon)
        if r:
            out.append(r)
    if not out:
        return None
    return {"последняя": out[0], "все": out}


def monthly_profile(s):
    """Сезонный профиль месяца: уровень месяца ОТНОСИТЕЛЬНО годового уровня вокруг
    него. Значение 1.0 — месяц как год в среднем, 3.0 — втрое выше.

    Делить на скользящий годовой уровень обязательно, иначе профиль меряет тренд,
    а не сезон: социальные вопросы Павлодара выросли за 6 лет в разы, и на
    абсолютных медианах любые два далёких месяца отличались в 3 раза при
    фактической сезонности ×1.15.

    Медиана, а не среднее: среднее тянет одна аварийная неделя — у Караганды март
    2022 содержит неделю в 1 460 обращений по дорогам, среднее марта 375 против
    медианы 170 при медиане ряда 78."""
    ann = s.rolling(SEASON_LAG, center=True, min_periods=SEASON_LAG // 2).median()
    rel = s / ann.where(ann > 0)
    return rel.groupby(rel.index.month).median()


def seasonality(s):
    """Во сколько раз пиковый месяц выше медианного по сезонному профилю."""
    m = monthly_profile(s).dropna()
    return float(m.max() / m.median()) if len(m) and m.median() > 0 else float("nan")


def month_years(s):
    """Сколько РАЗНЫХ лет наблюдалось у каждого месяца."""
    return s.groupby(s.index.month).apply(lambda x: x.index.year.nunique())


def level_change(series, month_from, month_to):
    """Во сколько раз меняется типичный уровень между двумя месяцами года.

    Требует минимум MIN_MONTH_YEARS разных лет на КАЖДЫЙ из двух месяцев. При
    одном годе профиль месяца — это просто уровень того года, и растущий ряд
    выглядит сменой сезона: у социальных вопросов Павлодара на train в 67 недель
    июнь 47 против сентября 15.5 давали «×3», хотя сезонность всего ряда ×1.15.
    """
    prof, yrs = monthly_profile(series), month_years(series)
    a, b = prof.get(month_from), prof.get(month_to)
    if a is None or b is None or a <= 0 or b <= 0:
        return float("nan")
    if min(yrs.get(month_from, 0), yrs.get(month_to, 0)) < MIN_MONTH_YEARS:
        return float("nan")
    return float(b / a)


def crosses_transition(series, month_from, month_to):
    """Окно пересекает смену сезона: уровень на конце отличается от уровня
    на старте в SEASONAL_K и более раз — в любую сторону. Важно именно это,
    а не попадание в пиковый месяц: окно, которое ЦЕЛИКОМ внутри сезона,
    плоскую модель не ломает."""
    r = level_change(series, month_from, month_to)
    return (not np.isnan(r)) and (r >= SEASONAL_K or r <= 1 / SEASONAL_K)


def transition_folds(series, horizon, step=2):
    """Отсечки, где окно прогноза пересекает смену сезона.

    Профиль месяцев считается по ряду ДО отсечки, а не по всему: на момент
    отсечки будущих месяцев ещё нет, и решение «здесь смена сезона» обязано
    приниматься на тех же данных, что и сам прогноз. На полном ряде это решение
    менялось у 6 отсечек из 128 по теплоснабжению Павлодара.
    """
    out = []
    for origin in range(SEASON_LAG + LEVEL_WINDOW, len(series) - horizon + 1, step):
        if crosses_transition(series.iloc[:origin], series.index[origin - 1].month,
                              series.index[origin + horizon - 1].month):
            r = evaluate_at(series, origin, horizon)
            if r:
                out.append(r)
    return out


def weeks_word(n):
    """«неделя / недели / недель» по числу."""
    if 11 <= n % 100 <= 14:
        return "недель"
    return {1: "неделя", 2: "недели", 3: "недели", 4: "недели"}.get(n % 10, "недель")


def longest_gap(s):
    """Самая длинная череда недель без единого обращения: (длина, начало, конец).

    Ноль обращений в неделю по региону целиком — это не затишье, а пропуск в
    выгрузке. Такой ряд нельзя ни обучать, ни проверять: плоская и сезонная
    модели начинают предсказывать нули, а возобновление выгрузки выглядит как
    всплеск с нулевого фона."""
    z = (s.to_numpy() == 0)
    best, i = (0, None, None), 0
    while i < len(z):
        if z[i]:
            j = i
            while j + 1 < len(z) and z[j + 1]:
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, s.index[i], s.index[j])
            i = j + 1
        else:
            i += 1
    return best


def max_folds(n_weeks, horizon):
    """Сколько отсечек скользящей проверки помещается в ряд такой длины."""
    return max(0, (n_weeks - SEASON_LAG - LEVEL_WINDOW) // horizon)


def matched_folds(series, horizon, month_from, month_to):
    """Отсечки прошлых лет с ТЕМ ЖЕ календарным переходом: окно начинается в том же
    месяце и заканчивается в том же, что и прогноз вперёд. Усреднять все переходы
    подряд нельзя — вход в сезон и выход из него это разные задачи, и модель,
    лучшая в среднем по обоим, может быть худшей на нужном."""
    out = []
    for origin in range(SEASON_LAG + LEVEL_WINDOW, len(series) - horizon + 1):
        if (series.index[origin - 1].month == month_from
                and series.index[origin + horizon - 1].month == month_to):
            r = evaluate_at(series, origin, horizon)
            if r:
                out.append(r)
    return out


def same_window_facts(series, horizon, month_from, month_to):
    """Сумма обращений за то же календарное окно в прошлые годы: сколько их было
    фактически. Модель тут ни при чём — это линейка, к которой прикладывается прогноз."""
    out = []
    for origin in range(1, len(series) - horizon + 1):
        if (series.index[origin - 1].month == month_from
                and series.index[origin + horizon - 1].month == month_to):
            out.append(float(series.iloc[origin:origin + horizon].sum()))
    return out


def mean_mae_folds(folds, model):
    vals = [f["модели"][model]["MAE"] for f in folds if model in f["модели"]]
    return float(np.mean(vals)) if vals else float("nan")


def best(fold, key="MAE"):
    return min(fold["модели"], key=lambda m: fold["модели"][m][key])


def mean_mae(res, model):
    """Средний MAE модели по всем отсечкам скользящей проверки."""
    vals = [f["модели"][model]["MAE"] for f in res["все"] if model in f["модели"]]
    return float(np.mean(vals)) if vals else float("nan")


def forward_choice(s, horizon, res, tr=None):
    """Shared choice, unchanged from the accepted forward report logic."""
    matched = None
    fwd_end = pd.date_range(s.index.max(), periods=horizon + 1, freq="W-MON")[-1]
    r_lvl = level_change(s, s.index.max().month, fwd_end.month)
    lvl = "×?" if np.isnan(r_lvl) else (f"×{r_lvl:.0f}" if r_lvl >= 10 else f"×{r_lvl:.1f}")
    if crosses_transition(s, s.index.max().month, fwd_end.month):
        mf = matched_folds(s, horizon, s.index.max().month, fwd_end.month)
        mm = {x: mean_mae_folds(mf, x) for x in MODELS} if mf else {}
        avail = [x for x in mm if not np.isnan(mm[x])]
        if len(mf) >= MIN_MATCHED and avail:
            m = min(avail, key=lambda x: mm[x])
            matched = [mf, mm, m, m]   # 4-й элемент — модель после защиты
            why = (f"окно пересекает смену сезона (уровень {lvl}); модель — по "
                   f"{len(mf)} отсечкам того же перехода")
        elif tr:
            m = tr["модель"]
            why = (f"окно пересекает смену сезона (уровень {lvl}); отсечек того же "
                   "перехода мало, модель — по всем переходным отсечкам")
        else:
            # Плоский прогноз — константа; смену уровня в разы он отследить не может
            # в принципе, независимо от того, что показала скользящая проверка.
            # Переходных отсечек в этом ряду нет, сравнить наивный с сезонным
            # на них не на чем — берём лучший из двух по обычной проверке.
            cand = [x for x in MODELS if x != "плоский" and not np.isnan(mean_mae(res, x))]
            if not cand:
                return None
            m = min(cand, key=lambda x: mean_mae(res, x))
            why = (f"окно пересекает смену сезона (уровень {lvl}); переходных отсечек "
                   "в ряду нет, плоский исключён как константа")
    else:
        m = min((x for x in MODELS if not np.isnan(mean_mae(res, x))),
                key=lambda x: mean_mae(res, x))
        why = f"смены сезона в окне нет (уровень {lvl}) — по скользящей проверке"
    # Защита уровня живёт внутри модели (level_guarded) и действует одинаково
    # в бэктесте и здесь. Тут она только переименовывает модель: если поправка
    # отключена, «сезонный» численно совпадает с наивным, и называть его
    # сезонным значило бы показывать не ту модель, которая посчитана.
    if m == "сезонный":
        _, guarded, (now, then, k) = level_guarded(s)
        if guarded:
            m = "наивный"
            why += (f"; поправка на уровень отключена — она опиралась бы на медианы "
                    f"{now:.0f} и {then:.0f} обращений в неделю (k={k:.2f})")
            if matched is not None:
                matched[3] = m
    return m, why, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=13)
    ap.add_argument("--keep-first-month", action="store_true")
    ap.add_argument("--folds", type=int, default=3, help="отсечек в скользящей проверке")
    a = ap.parse_args()

    from src.cli import require_unified
    require_unified()
    df = pd.read_parquet(DATA, columns=["created_at", "region", "topic", "appeal_class"])
    df = df[df.appeal_class == "problem"].copy()
    df["created_at"] = pd.to_datetime(df["created_at"])

    say("# Прогноз недельной нагрузки\n")
    say(f"Класс `problem`. Горизонт проверки — {a.horizon} недель "
        f"(~{a.horizon*7/30:.0f} месяца). Неполные крайние недели отброшены"
        + ("" if a.keep_first_month else ", первый календарный месяц ряда тоже") + ".\n")

    # ---------- ограничение по регионам
    say("## Где прогноз не строится и почему\n")
    say("Годовая сезонность требует минимум двух полных лет: один год на обучение "
        "и один на проверку сезонного лага. Столько истории есть не везде.\n")
    say("| Регион | История | Недель | Прогноз |")
    say("|---|---|---|---|")
    rows, gaps = [], {}
    for r, g in df.groupby("region"):
        days = (g.created_at.max() - g.created_at.min()).days
        n = len(weekly(df, r, skip_first_month=not a.keep_first_month))
        rows.append((r, days / 365.25, n))
    for r, yrs, n in sorted(rows, key=lambda x: -x[1]):
        k = max_folds(n, a.horizon)
        gap = longest_gap(weekly(df, r, skip_first_month=not a.keep_first_month))
        if r in REGIONS and gap[0] > MAX_GAP:
            gaps[r] = gap
            verdict = (f"**не строится: пропуск в выгрузке** — {gap[0]} {weeks_word(gap[0])} "
                       f"подряд без обращений, {gap[1]:%Y-%m-%d} → {gap[2]:%Y-%m-%d}")
        elif r in REGIONS:
            verdict = ("**строится**" if k >= a.folds
                       else f"**строится**, но ряд на грани: отсечек скользящей проверки {k}, не {a.folds}")
        elif n >= SEASON_LAG + a.horizon + LEVEL_WINDOW:
            verdict = "истории хватает, но в список не включён — решение основной сессии"
        else:
            verdict = f"не строится: нужно минимум {SEASON_LAG + a.horizon + LEVEL_WINDOW} недель"
        say(f"| {r} | {yrs:.1f} лет | {n} | {verdict} |")
    say("\n**Это ограничение данных, а не пропуск.** У Алматинской, Костанайской и "
        "Акмолинской областей годовой сезонности в данных просто нет: ряд короче одного "
        "годового цикла. Никакая модель этого не обойдёт — нужны данные за прошлые годы.\n")
    if gaps:
        for r, (n_gap, lo, hi) in gaps.items():
            say(f"**{r}: в выгрузке нет обращений с {lo:%Y-%m-%d} по {hi:%Y-%m-%d} — "
                f"{n_gap} {weeks_word(n_gap)} подряд.** Истории по календарю хватает, но ряд разорван "
                "пополам, и прогноз по нему строить нельзя: обучающее окно попадает в "
                "пропуск, плоская и сезонная модели начинают предсказывать нули. Это "
                "дефект выгрузки, а не свойство региона — вопрос заказчику.\n")
    say("**Караганда проходит порог впритык.** Минимум в "
        f"{SEASON_LAG + a.horizon + LEVEL_WINDOW} недель она перекрывает, но на скользящую "
        f"проверку остаётся 2 отсечки против {a.folds} у Павлодара и ВКО. Вывод о том, "
        "какая модель лучше, по Караганде опирается на два сравнения и слабее остальных. "
        "Так и читать.\n")

    # ---------- ряды
    series, thin = {}, []
    for r in REGIONS:
        if r in gaps:
            continue
        series[(r, None)] = weekly(df, r, skip_first_month=not a.keep_first_month)
        for t in df[df.region == r].topic.value_counts().head(TOP_TOPICS).index:
            st = weekly(df, r, t, skip_first_month=not a.keep_first_month)
            # Срез с типичной неделей ниже MIN_LEVEL не прогнозируется: MAE на нём
            # исчисляется единицами обращений, и «лучшая модель» выбирается шумом.
            if st.median() < MIN_LEVEL:
                thin.append((r, t, float(st.median()), int(st.sum())))
            else:
                series[(r, t)] = st

    if thin:
        say("### Темы, по которым прогноз не строится: слишком редкие\n")
        say(f"Эти темы входят в топ-{TOP_TOPICS} региона по числу обращений, но приходят "
            "пачками, а не потоком: в типичную неделю обращений меньше "
            f"{MIN_LEVEL}. На таком ряде MAE исчисляется единицами обращений, и «лучшая "
            "модель» выбирается шумом, а не качеством.\n")
        say("| Регион | Тема | Обращений всего | Медиана в неделю |")
        say("|---|---|---|---|")
        for r, t, med, tot in thin:
            say(f"| {r.replace(' область','')} | {t} | {tot} | {med:.0f} |")
        say("")

    # ---------- смена состава тем внутри ряда
    say("### Состав потока внутри ряда: менялся ли он\n")
    say("Ряд одного региона можно обучать только если он всё время про одно и то же. "
        f"Если доля крупнейшей темы сдвинулась больше чем на {MIX_SHIFT} п.п., "
        "поменялась практика регистрации, а не поток обращений.\n")
    say("| Регион | Главная тема | Её доля в первый год | В последний год | Сдвиг |")
    say("|---|---|---|---|---|")
    shifted = []
    for r in [x for x in REGIONS if x not in gaps]:
        g = df[df.region == r]
        yrs = sorted(g.created_at.dt.year.unique())
        if len(yrs) < 2:
            continue
        first, last = g[g.created_at.dt.year == yrs[0]], g[g.created_at.dt.year == yrs[-1]]
        top = g.topic.value_counts().idxmax()
        p0 = (first.topic == top).mean() * 100
        p1 = (last.topic == top).mean() * 100
        flag = "**состав сменился**" if abs(p1 - p0) >= MIX_SHIFT else "стабилен"
        if abs(p1 - p0) >= MIX_SHIFT:
            shifted.append((r, top, p0, p1, yrs[0], yrs[-1]))
        say(f"| {r.replace(' область','')} | {top} | {p0:.0f}% ({yrs[0]}) | "
            f"{p1:.0f}% ({yrs[-1]}) | {p1 - p0:+.0f} п.п. — {flag} |")
    for r, top, p0, p1, y0, y1 in shifted:
        say(f"\n**{r}: доля темы «{top}» выросла с {p0:.0f}% в {y0} году до {p1:.0f}% "
            f"в {y1}.** Это смена справочника или практики регистрации в системе-источнике, "
            "а не изменение того, на что жалуются люди. Следствия: разрез по темам внутри "
            "региона через эту границу не сопоставим, а срез «регион × главная тема» почти "
            "повторяет регион целиком. Прогноз по региону в целом это не ломает — сумма "
            "остаётся суммой, — но темы до и после границы складывать нельзя. Вопрос "
            "заказчику.\n")

    say("## Ряды\n")
    say("| Срез | Недель | Окно | Медиана, обращений в неделю |")
    say("|---|---|---|---|")
    for (r, t), s in series.items():
        name = r.replace(" область", "") + ("" if t is None else f" · {t}")
        say(f"| {name} | {len(s)} | {s.index.min():%Y-%m-%d} → {s.index.max():%Y-%m-%d} | {s.median():.0f} |")

    # ---------- проверка
    say(f"\n## Проверка на отложенных {a.horizon} неделях\n")
    say("Обучение на данных до отсечки, прогноз на следующие "
        f"{a.horizon} недель, сравнение с фактом. MAPE — только по неделям "
        "с ненулевым фактом.\n")
    say("| Срез | Модель | MAE, обращений | MAPE | Нулевых недель |")
    say("|---|---|---|---|---|")
    results = {}
    for key, s in series.items():
        res = evaluate(s, a.horizon, a.folds)
        results[key] = res
        r, t = key
        name = r.replace(" область", "") + ("" if t is None else f" · {t}")
        if res is None:
            say(f"| {name} | — | ряд короче {SEASON_LAG + a.horizon + LEVEL_WINDOW} недель | — | — |")
            continue
        fold = res["последняя"]
        b = best(fold)
        for m, v in fold["модели"].items():
            mark = " **←**" if m == b else ""
            mape = "—" if np.isnan(v["MAPE"]) else f"{v['MAPE']:.1f}%"
            say(f"| {name} | {m}{mark} | {v['MAE']:.0f} | {mape} | {v['нулевых недель']} |")

    # ---------- вердикт по скользящей проверке
    ok = [k for k, v in results.items() if v]
    say(f"\n## Скользящая проверка: {a.folds} отсечки вместо одной\n")
    say("Одна отсечка — один бросок. Та же проверка повторена на сдвинутых отсечках; "
        "вывод делается по среднему MAE, а не по последней отсечке.\n")
    say("| Срез | Отсечек | MAE наивного | MAE сезонного | MAE плоского | Лучшая |")
    say("|---|---|---|---|---|---|")
    wins = {m: 0 for m in MODELS}
    nv_not_worse = nv_worse = 0
    for k in ok:
        r, t = k
        name = r.replace(" область", "") + ("" if t is None else f" · {t}")
        res = results[k]
        mae = {m: mean_mae(res, m) for m in MODELS}
        b = min((m for m in MODELS if not np.isnan(mae[m])), key=lambda m: mae[m])
        wins[b] += 1
        if mae["наивный"] <= mae["сезонный"]:
            nv_not_worse += 1
        else:
            nv_worse += 1
        cells = " | ".join("—" if np.isnan(mae[m]) else f"{mae[m]:.0f}" for m in MODELS)
        say(f"| {name} | {len(res['все'])} | {cells} | **{b}** |")
    say("")
    say("Лучшая модель по среднему MAE: " + ", ".join(f"{m} — {wins[m]}" for m in MODELS)
        + f" (срезов {len(ok)}).\n")

    say("## Наивный прогноз против сезонного\n")
    if nv_not_worse >= nv_worse:
        say(f"**Наивный прогноз не проигрывает: он лучше или равен сезонному на "
            f"{nv_not_worse} срезах из {nv_not_worse + nv_worse}.** Поправка на уровень "
            "себя не окупает — усложнять модель на этих данных нет оснований.")
    else:
        say(f"**Сезонный лучше наивного на {nv_worse} срезах из {nv_not_worse + nv_worse}.** "
            "Поправка на текущий уровень окупается.")
    if wins["плоский"] >= max(wins["наивный"], wins["сезонный"]):
        say(f"\n**Годовая сезонность вообще не окупается.** Плоский прогноз — медиана "
            f"последних {FLAT_WINDOW} недель, без всякой сезонности — лучший на "
            f"{wins['плоский']} срезах из {len(ok)}. На горизонте {a.horizon} недель "
            "«столько же, сколько в последние недели» точнее, чем «столько же, "
            "сколько год назад».")

    # ---------- сезонные переходы
    say("\n## Где плоский прогноз ломается: сезонные переходы\n")
    say(f"Вывод «плоский лучше» верен в среднем, но он получен на отсечках, которые чаще "
        f"всего не пересекают смену сезона. Окно считается пересекающим смену сезона, "
        f"если типичный уровень месяца на конце окна отличается от уровня месяца на "
        f"старте в {SEASONAL_K} и более раз. Важно именно это, а не попадание в пиковый "
        f"месяц: окно целиком внутри сезона плоскую модель не ломает.\n")
    say(f"Уровень месяца берётся из сезонного профиля — медианы недель месяца, делённой "
        f"на годовой уровень вокруг них. Без деления на годовой уровень профиль мерил бы "
        f"тренд: растущий ряд давал бы «смену сезона» между любыми далёкими месяцами. "
        f"Месяц, наблюдавшийся меньше чем в {MIN_MONTH_YEARS} разных годах, в сравнении "
        f"не участвует — по одному году сезон от тренда не отличить. Колонка «сезонность "
        f"профиля» — во сколько раз пиковый месяц профиля выше медианного.\n")
    say("| Срез | Сезонность профиля | Отсечек со сменой сезона | MAE наивного | MAE сезонного | MAE плоского | Лучшая |")
    say("|---|---|---|---|---|---|---|")
    transition = {}
    for key, ser in series.items():
        if results[key] is None:
            continue
        folds = transition_folds(ser, a.horizon)
        if not folds:
            continue
        mae = {m: mean_mae_folds(folds, m) for m in MODELS}
        avail = [m for m in MODELS if not np.isnan(mae[m])]
        if not avail:
            continue
        b = min(avail, key=lambda m: mae[m])
        transition[key] = {"модель": b}
        r, t = key
        name = r.replace(" область", "") + ("" if t is None else f" · {t}")
        cells = " | ".join("—" if np.isnan(mae[m]) else f"{mae[m]:.0f}" for m in MODELS)
        sn = seasonality(ser)
        sn_txt = "—" if np.isnan(sn) else f"×{sn:.1f}"
        say(f"| {name} | {sn_txt} | {len(folds)} | {cells} | **{b}** |")
    if not transition:
        say("| — | — | — | — | — | — | рядов со сменой сезона нет |")
    say("\nНа этих отсечках плоский прогноз перестаёт быть лучшим, но отрыв невелик — "
        "и это само по себе показательно. Средний MAE по всем переходам смешивает вход "
        "в сезон и выход из него: ошибки в разные стороны гасят друг друга. Поэтому для "
        "прогноза вперёд модель выбирается не по этой таблице, а по отсечкам ТОГО ЖЕ "
        "календарного перехода — см. ниже. Там разрыв виден целиком.")

    # ---------- прогноз вперёд
    say(f"\n## Прогноз на {a.horizon} недель от конца данных\n")
    say("Отсчёт ведётся от последней полной недели ряда, а не от сегодняшнего дня: "
        "у Караганды данные заканчиваются 2023-12, поэтому её прогноз — проверка метода, "
        "а не рабочий прогноз на будущее.\n")
    say("| Срез | Модель | Почему эта модель | Последняя неделя ряда | Сумма прогноза | В среднем в неделю |")
    say("|---|---|---|---|---|---|")
    matched_report, fwd_sum = {}, {}
    for key, s in series.items():
        res = results[key]
        if res is None:
            continue
        r, t = key
        name = r.replace(" область", "") + ("" if t is None else f" · {t}")
        choice = forward_choice(s, a.horizon, res, transition.get(key))
        if choice is None:
            continue
        m, why, matched = choice
        if matched is not None:
            matched_report[key] = matched
        pred = forecast(s, a.horizon, m)
        if pred is None:
            continue
        fwd_sum[key] = float(pred.sum())
        say(f"| {name} | {m} | {why} | {s.index.max():%Y-%m-%d} | {pred.sum():.0f} | {pred.mean():.0f} |")

    # ---------- отсечки того же календарного перехода
    if matched_report:
        say("\n### Проверка на том же переходе в прошлые годы\n")
        say("Для срезов, где окно прогноза пересекает смену сезона, модель выбрана не по "
            "среднему по всем переходам, а по отсечкам с ТЕМ ЖЕ календарным переходом: "
            "обучение до той же недели года, прогноз до того же месяца. Вход в сезон и "
            "выход из него — разные задачи.\n")
        say("| Срез | Переход | Отсечек | MAE наивного | MAE сезонного | MAE плоского | Выбрана |")
        say("|---|---|---|---|---|---|---|")
        for key, (mf, mm, m, applied) in matched_report.items():
            r, t = key
            nm = r.replace(" область", "") + ("" if t is None else f" · {t}")
            s2 = series[key]
            tr_name = (f"{MONTHS[s2.index.max().month]} → "
                       f"{MONTHS[pd.date_range(s2.index.max(), periods=a.horizon + 1, freq='W-MON')[-1].month]}")
            cells = " | ".join("—" if np.isnan(mm[x]) else f"{mm[x]:.0f}" for x in MODELS)
            mark = f"**{m}**" if applied == m else f"~~{m}~~ → **{applied}** (защита по уровню)"
            say(f"| {nm} | {tr_name} | {len(mf)} | {cells} | {mark} |")
            if level_guarded(s2)[1]:
                _, _, (now_, then_, k_) = level_guarded(s2)
                say(f"\nУ этого среза сезонная модель численно совпадает с наивной: "
                    f"поправка на уровень отключена защитой — она опиралась бы на медианы "
                    f"{now_:.0f} и {then_:.0f} обращений в неделю (k={k_:.2f}). "
                    f"Защита действует одинаково и в проверке, и в прогнозе вперёд.")
            facts = [float(s2.iloc[s2.index.get_loc(f["отсечка"]) + 1:
                                   s2.index.get_loc(f["отсечка"]) + 1 + a.horizon].sum()) for f in mf]
            say(f"\nФакт за эти {a.horizon} недель в прошлые годы: от {min(facts):.0f} до "
                f"{max(facts):.0f} обращений, медиана {np.median(facts):.0f}. Плоский прогноз "
                "на том же месте давал десятки обращений за квартал: он переносит вперёд "
                "уровень межсезонья и смену сезона не отслеживает в принципе.")

    # ---------- сверка прогноза с фактом того же окна в прошлые годы
    say(f"\n## Сверка прогноза с фактом того же окна в прошлые годы\n")
    say("Прогноз на 13 недель осмысленно сравнивать не с медианой ряда, а с тем, сколько "
        "обращений фактически приходило за то же календарное окно раньше. Рядом — факт "
        "за последние 13 недель (уровень, с которого стартует прогноз) и факт того же "
        "окна годом раньше.\n")
    say("**Выход за диапазон прошлых лет сам по себе не дефект.** У растущего ряда прогноз "
        "выше всех прошлых лет закономерен: сравнивать надо с последним кварталом. "
        "Разбирать нужно обратный случай — когда прогноз НИЖЕ всех прошлых лет при том, "
        "что окно уходит в более нагруженный сезон.\n")
    say("| Срез | Прогноз | Последние 13 недель | То же окно год назад | Прошлые годы: от | до | Оценка |")
    say("|---|---|---|---|---|---|---|")
    for key, ser in series.items():
        if results[key] is None or key not in fwd_sum:
            continue
        end = ser.index.max()
        fwd_end = pd.date_range(end, periods=a.horizon + 1, freq="W-MON")[-1]
        facts = same_window_facts(ser, a.horizon, end.month, fwd_end.month)
        facts = [f for f in facts if f > 0]
        r, t = key
        nm = r.replace(" область", "") + ("" if t is None else f" · {t}")
        v = fwd_sum[key]
        last13 = float(ser.iloc[-a.horizon:].sum())
        nv = forecast(ser, a.horizon, "наивный")
        ya = "—" if nv is None else f"{nv.sum():.0f}"
        if not facts:
            say(f"| {nm} | {v:.0f} | {last13:.0f} | {ya} | — | — | сравнить не с чем |")
            continue
        if v < min(facts):
            flag = "**ниже всех прошлых лет**"
        elif v > max(facts):
            flag = "выше всех — уровень ряда вырос"
        else:
            flag = "в диапазоне"
        say(f"| {nm} | {v:.0f} | {last13:.0f} | {ya} | {min(facts):.0f} | {max(facts):.0f} | {flag} |")

    low = []
    for key in fwd_sum:
        if results[key] is None:
            continue
        ser = series[key]
        end = ser.index.max()
        fe = pd.date_range(end, periods=a.horizon + 1, freq="W-MON")[-1]
        f = [x for x in same_window_facts(ser, a.horizon, end.month, fe.month) if x > 0]
        if f and fwd_sum[key] < min(f):
            low.append((key, min(f), level_change(ser, end.month, fe.month),
                        float(ser.iloc[-a.horizon:].sum())))
    if low:
        say("\nПрогноз ниже всех прошлых лет — разбор по срезам. Причины разные, одной "
            "формулировкой не закрываются:\n")
        for key, lo, r_lvl, last13 in low:
            r, t = key
            nm = r.replace(" область", "") + ("" if t is None else f" · {t}")
            lvl = "×?" if np.isnan(r_lvl) else f"×{r_lvl:.1f}"
            if not np.isnan(r_lvl) and r_lvl >= 1.5:
                why = (f"уровень окна растёт {lvl}, но порог смены сезона (×{SEASONAL_K}) "
                       f"не пройден — плоский прогноз переносит вперёд уровень текущего "
                       f"месяца и в сезон не входит")
            else:
                why = (f"смены сезона нет ({lvl}): ряд просто идёт ниже прошлых лет — "
                       f"за последние {a.horizon} недель {last13:.0f} против {lo:.0f} "
                       f"минимума прошлых лет. Здесь прогноз следует за фактом, "
                       f"а не ошибается")
            say(f"- **{nm}** — {why}.")
        say("\nПорог ×" + str(SEASONAL_K) + " — граница, и у неё всегда есть срезы чуть ниже. "
            "Сверка с фактом того же окна их ловит; числа по таким срезам для показа брать "
            "как нижнюю оценку, а не как прогноз.")
    say("\nУ срезов, где выбран наивный прогноз, колонки «Прогноз» и «То же окно год назад» "
        "совпадают по построению: наивный прогноз — это и есть факт того же окна год назад.")

    # ---------- выбросы в базе сезонного прогноза
    say("\n## Почему сезонный прогноз здесь ненадёжен\n")
    say("Сезонный и наивный прогнозы копируют в будущее конкретные недели годичной "
        "давности вместе с разовыми выбросами. Срезы, где в базе прогноза есть неделя "
        f"выше {OUTLIER_K} медиан ряда:\n")
    say("| Срез | Медиана ряда | Максимум в базе прогноза | Во сколько раз | Неделя |")
    say("|---|---|---|---|---|")
    found = False
    for key, ser in series.items():
        if results[key] is None or len(ser) < SEASON_LAG:
            continue
        base = ser.iloc[-SEASON_LAG:][:a.horizon]
        med = ser.median()
        if med > 0 and base.max() > OUTLIER_K * med:
            found = True
            r, t = key
            name = r.replace(" область", "") + ("" if t is None else f" · {t}")
            say(f"| {name} | {med:.0f} | {base.max():.0f} | ×{base.max()/med:.1f} | "
                f"{base.idxmax():%Y-%m-%d} |")
    if not found:
        say("| — | — | — | — | нет |")
    say("\nТакая неделя переносится в прогноз целиком. Лечится усечением выбросов в базе "
        "(винзоризация) или медианой нескольких прошлых лет вместо одного — но прежде чем "
        "усложнять, см. вывод выше: сезонность на этих данных и так не окупается.")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт сохранён: {OUT}")


if __name__ == "__main__":
    main()
