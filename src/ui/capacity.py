"""Planning calculator presentation, isolated from the other dashboard blocks."""
from src.capacity_scenario import HORIZON_WEEKS, MAX_RESERVE_PCT, shift_capacity, utilization_status
from src.ui.components import esc, number, status_badge, section_header
from src.ui.theme import GREEN, AMBER, GREY

DISCLAIMER = ('Параметры работы операторов введены пользователем. '
              'Расчёт является сценарной оценкой мощности, а не рекомендацией по штатной численности.')
SCENARIO_NOTE = ('−20% и +20% — сценарии изменения входящего потока, '
                 'а не доверительный интервал прогноза.')
SCENARIOS = [('−20%', 'Сниженный поток'), ('Базовый', 'Ожидаемый поток'), ('+20%', 'Повышенный поток')]

CSS = '''<style>
.nazar-capacity-card{background:white;border:1px solid #d8e2ee;border-radius:8px;padding:20px;min-height:410px;color:#16324f}
.nazar-capacity-card .cap-title{font-size:22px;margin:0!important;padding:0!important;color:#16324f}
.nazar-capacity-card .cap-sub{font-size:13px;color:#596b7c;margin:4px 0 16px}
.nazar-capacity-card dl{margin:12px 0}.nazar-capacity-card dt{font-size:13px;color:#596b7c;margin-top:12px}
.nazar-capacity-card dd{margin:4px 0 0;font-size:25px;font-weight:650;line-height:1.2}
.nazar-capacity-card dd small{display:inline;font-size:12px;font-weight:400;margin-left:6px;color:#596b7c}
.nazar-capacity-card .cap-balance{border-top:1px solid #e3e9ef;padding-top:12px;margin-top:12px}
.nazar-capacity-note{border:1px solid #cfe2e5;background:#e9f3f7;color:#16324f;padding:10px 14px;border-radius:8px;font-size:13px;line-height:1.5;margin-bottom:8px}
@media(max-width:1400px){.nazar-capacity-card{padding:16px}.nazar-capacity-card dd{font-size:23px}}
.st-key-capacity_parameters [data-testid="stVerticalBlock"]{gap:8px}
</style>'''


def scenario_card(st, index, row=None):
    title, subtitle = SCENARIOS[index]
    label, tone = utilization_status(row['utilization']) if row else ('Заполните параметры', 'neutral')
    color = (GREEN if tone == 'ready' else AMBER) if row else GREY
    demand = number(row['demand']) if row else '—'
    available = number(row['available_capacity']) if row else '—'
    utilization = number(row['utilization'] * 100, 1) + ' %' if row else '—'
    balance_label = 'Дефицит' if row and row['balance'] < 0 else 'Резерв'
    balance = ('−' if row['balance'] < 0 else '+') + number(abs(row['balance'])) if row else '—'
    st.markdown(f'<article class="nazar-capacity-card" style="--accent:{color}">'
                f'<div class="cap-title">{esc(title)}</div><div class="cap-sub">{esc(subtitle)}</div>'
                f'{status_badge(label,tone)}<dl><dt>Ожидаемый поток</dt><dd>{esc(demand)}'
                f'<small>обращений</small></dd>'
                f'<dt>Доступная мощность</dt><dd>{esc(available)}<small>обращений</small></dd>'
                f'<dt>Загрузка</dt><dd>{esc(utilization)}</dd>'
                f'<dt class="cap-balance">{balance_label}</dt><dd>{esc(balance)}<small>обращений</small></dd>'
                '</dl></article>',unsafe_allow_html=True)


def capacity_section(st, forecasts):
    section_header(st, 'Сценарий нагрузки',
                   'Оцените, хватит ли заданной мощности команды при ожидаемом потоке обращений.', anchor='capacity-scenario')
    st.html(CSS)
    st.markdown(f'<div class="nazar-capacity-note">{esc(DISCLAIMER)}<br>{esc(SCENARIO_NOTE)}</div>',unsafe_allow_html=True)
    if not forecasts:
        st.info('Нет доступного прогноза за 13 недель. Сравнение мощности с потоком пока недоступно.')
        return
    index = st.selectbox('Ряд для сценария', range(len(forecasts)), key='cap_forecast',
                        format_func=lambda i: forecasts[i]['region']+' · '+(forecasts[i]['topic'] or 'все темы'))
    selected = forecasts[index]
    if selected['weeks'] != HORIZON_WEEKS:
        st.warning('Для сравнения нужен прогноз за те же 13 недель.')
        return
    left, right = st.columns([1, 3], gap='medium')
    with left.container(border=True, key="capacity_parameters"):
        st.markdown('**Параметры работы**')
        operators = st.number_input('Операторов в смене', min_value=1, value=None, step=1, key='cap_operators')
        hours = st.number_input('Часов в смене', min_value=0., max_value=24., value=None, key='cap_hours')
        shifts = st.number_input('Смен в неделю', min_value=0., value=None, key='cap_shifts',
                                 help='Число смен команды в неделю; можно указать среднее значение.')
        aht = st.number_input('Среднее время обработки, мин', min_value=0., value=None, key='cap_aht')
        reserve = st.number_input('Резерв мощности, %', min_value=0., max_value=MAX_RESERVE_PCT,
                                  value=None, key='cap_reserve', help='0–95% — границы сценарной формы, не норматив 109.')
    rows = None
    with right:
        if all(v is not None for v in (operators, hours, shifts, aht, reserve)):
            try:
                rows = shift_capacity(selected['demand'], operators, hours, shifts, aht, reserve,
                                      forecast_weeks=selected['weeks'])
            except ValueError as error:
                st.warning(str(error))
        else:
            st.caption('Заполните все пять полей. Значения не заданы за вас.')
        columns = st.columns(3)
        for i, column in enumerate(columns):
            scenario_card(column, i, rows[i] if rows else None)
        st.caption(f"Одинаковый горизонт: {HORIZON_WEEKS} недель · данные по {selected['data_as_of']} · прогноз до {selected['horizon_end']}.")
        if rows:
            st.caption('Мощность = операторы × часы в смене × смены в неделю × 13 × 60 / время обработки × (1 − резерв / 100).')
