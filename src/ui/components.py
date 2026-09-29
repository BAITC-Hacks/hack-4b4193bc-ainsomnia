"""Escaped presentation of existing aggregates; no scoring or detection."""
import html
from datetime import date

from src.ui.theme import NAVY, TEAL, GREEN, AMBER, RED, GREY

PAGES = (
    ('operations', 'Оперативно'),
    ('planning', 'Планирование'),
    ('control', 'Контроль'),
    ('analytics', 'Аналитика'),
    ('reports', 'Отчёты'),
)

TYPES = {'NEW_SPIKE':'Новый всплеск','SLA_RISK':'Риск просрочки',
         'DATA_QUALITY':'Качество данных','DATA_STALE':'Данные требуют обновления',
         'FORECAST_CHANGE':'Изменение нагрузки'}
PRIORITY = {'HIGH':('Высокий приоритет','attention',AMBER),
            'WATCH':('Наблюдение','neutral',TEAL),
            'BLOCKED':('Использовать нельзя','blocked',RED)}


def esc(value):
    return html.escape(str(value if value is not None else '—'), quote=True)


def number(value, digits=0):
    if value is None:
        return '—'
    return f'{value:,.{digits}f}'.replace(',', ' ').replace('.', ',')


def day(value):
    if not value:
        return '—'
    try:
        return date.fromisoformat(str(value)[:10]).strftime('%d.%m.%Y')
    except ValueError:
        return str(value)


def status_badge(label, tone='neutral'):
    tone = tone if tone in ('attention','ready','blocked','neutral') else 'neutral'
    return f'<span class="nazar-badge {tone}">{esc(label)}</span>'


def section_header(st, title, note=None, anchor=None):
    st.subheader(title, anchor=anchor)
    if note:
        st.caption(note)


def page_header(st, title, subtitle, df, source, presentation=False):
    """Compact page chrome; the full product identity lives in the sidebar."""
    source_label = 'ТЕСТОВЫЕ ДАННЫЕ' if str(source).lower() == 'fake' else 'REAL'
    st.markdown(
        f'<header class="nazar-page-head"><div><h1>{esc(title)}</h1>'
        f'<p>{esc(subtitle)}</p></div><div class="nazar-page-meta">'
        f'<span class="nazar-source {"fake" if str(source).lower() == "fake" else "real"}">'
        f'{esc(source_label)}</span><span>Данные по {df.region.nunique()} регионам</span>'
        f'<span>обновлено до {esc(day(df.created_at.max().date()))}</span></div></header>',
        unsafe_allow_html=True,
    )
    if presentation:
        st.button('Показать навигацию', key='exit_presentation', type='secondary',
                  on_click=lambda: st.session_state.update(presentation=False))


def sidebar_nav(st, df, source):
    """Persistent product navigation backed by session_state, not URL routing."""
    if st.session_state.get('page') not in dict(PAGES):
        st.session_state['page'] = 'operations'
    current = st.session_state['page']

    def choose(page):
        st.session_state['page'] = page

    with st.sidebar:
        st.markdown('<div class="nazar-side-brand">NAZAR-109</div>'
                    '<div class="nazar-side-sub">Аналитика обращений 109</div>'
                    '<div class="nazar-side-rule"></div>', unsafe_allow_html=True)
        for key, label in PAGES:
            st.button(label, key=f'nav_{key}', width='stretch',
                      type='primary' if current == key else 'secondary',
                      on_click=choose, args=(key,))
        st.markdown('<div class="nazar-side-rule"></div>', unsafe_allow_html=True)
        if str(source).lower() == 'fake':
            st.markdown('<div class="nazar-side-fake">ТЕСТОВЫЕ ДАННЫЕ</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="nazar-side-source">Источник <b>REAL</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="nazar-side-facts"><b>{df.region.nunique()} регионов</b>'
                    f'<b>{esc(number(len(df)))} записей</b><span>Последнее обновление</span>'
                    f'<b>{esc(day(df.created_at.max().date()))}</b></div>', unsafe_allow_html=True)
        st.markdown('<div class="nazar-side-rule"></div>', unsafe_allow_html=True)
        st.toggle('Режим презентации', key='presentation',
                  help='Скрывает навигацию и технические пояснения; данные не меняются.')
    return current


def metric_card(st, label, value, note='', color=TEAL):
    st.markdown(f'<div class="nazar-metric" style="--accent:{esc(color)}">'
                f'<div class="nazar-metric-label">{esc(label)}</div>'
                f'<div class="nazar-metric-value">{esc(value)}</div>'
                f'<div class="nazar-metric-note">{esc(note)}</div></div>',unsafe_allow_html=True)


def empty_state(st, message='По выбранным фильтрам данных нет.'):
    st.markdown(f'<div class="nazar-empty">{esc(message)}</div>',unsafe_allow_html=True)


def info_callout(st, message):
    st.markdown(f'<div class="nazar-callout">{esc(message)}</div>',unsafe_allow_html=True)


def masthead(st, df, source):
    st.markdown('<header class="nazar-header"><div><div class="nazar-brand">NAZAR109</div>'
                '<div class="nazar-brand-sub">Аналитика обращений 109</div></div>'
                '<div class="nazar-header-meta">'
                f'<span>Источник<b>{esc(source.upper())}</b></span>'
                f'<span>Последняя дата данных<b>{day(df.created_at.max().date())}</b></span>'
                f'<span>Регионов<b>{df.region.nunique()}</b></span>'
                f'<span>Записей<b>{number(len(df))}</b></span></div></header>',unsafe_allow_html=True)
    st.caption('Қазақстан · 109   |   У регионов разные периоды наблюдений. Даты относятся к выгрузке, не к сегодня.')


def fake_banner(st):
    st.markdown('<div class="nazar-fake" role="status"><strong>ТЕСТОВЫЕ ДАННЫЕ</strong>'
                '<span>ПОДДЕЛЬНЫЕ ДАННЫЕ — числа не описывают реальные регионы.</span></div>',unsafe_allow_html=True)


def executive_summary(st, snapshot):
    risk=snapshot['risk']; actions=snapshot['actions']
    columns=st.columns(4)
    metric_card(columns[0],'Обращений',number(snapshot['summary']['row_count']),'Все записи в проверенной выгрузке',NAVY)
    metric_card(columns[1],'Требует внимания',number(len(actions)),'Конкретных следующих проверок',AMBER)
    metric_card(columns[2],'Новых всплесков',number(snapshot['new_spikes']),'За последние 7 дней данных регионов',TEAL)
    metric_card(columns[3],'В очереди риска',number(risk.get('n')),
                'Карагандинская область · исторический тест' if risk['available'] else 'Расчёт недоступен',TEAL)


def action_card(st, item):
    label,tone,color=PRIORITY.get(item['severity'],('Требует внимания','attention',AMBER))
    e=item['evidence'];kind=item['type'];value='Проверить данные';note=item['headline']
    if kind=='NEW_SPIKE':
        value='+'+number(e['excess']);note=f"жалоб сверх обычного · пик {number(e['count'])}, обычно {number(e['baseline'],1)}"
    elif kind=='SLA_RISK' and 'n' in e:
        value=number(e['n']);note='обращений для приоритетной проверки'
    elif kind=='FORECAST_CHANGE':
        value=number(e.get('demand'));note=f"жалоб за {e.get('weeks','—')} недель · смена сезона"
    st.markdown(f'<article class="nazar-action" style="--accent:{color}"><div>{status_badge(label,tone)}'
                f'<div class="nazar-action-region">{esc(item["region"] or "Все регионы")}</div>'
                f'<div class="nazar-action-note">{esc(item["topic"] or "Все темы")}</div></div>'
                f'<div><div class="nazar-action-title">{esc(TYPES.get(kind,"Проверка"))}</div>'
                f'<div class="nazar-action-note">{esc(note)}</div></div><div>'
                f'<div class="nazar-action-value">{esc(value)}</div>'
                f'<div class="nazar-action-note">Данные по {esc(day(item["data_as_of"]))}</div></div></article>',unsafe_allow_html=True)


def spike_card(st, item):
    e = item['evidence']
    st.markdown(
        f'<article class="nazar-spike"><div class="nazar-spike-meta">'
        f'<b>{esc(item["region"])}</b><span>{esc(item["topic"] or "Все темы")}</span>'
        f'<span>{esc(day(item["event_date"]))}</span></div>'
        f'<div class="nazar-spike-value">+{esc(number(e.get("excess")))}</div>'
        f'<div class="nazar-spike-label">обращений сверх обычного</div>'
        f'<div class="nazar-spike-foot"><span>Пик <b>{esc(number(e.get("count")))}</b></span>'
        f'<span>Обычно <b>{esc(number(e.get("baseline"), 1))}</b></span></div></article>',
        unsafe_allow_html=True,
    )


def data_health_card(st, region, item):
    label,tone=('Данные готовы','ready') if item['status']=='OK' else ('Есть ограничения','attention')
    if item['status']=='BLOCKED':label,tone='Использовать нельзя','blocked'
    gaps=item['gaps_ge7']
    historical = 'Караганд' in region and item.get('forecast', {}).get('reason', '').lower().find('истор') >= 0
    if historical:
        label, tone = 'Исторические данные', 'neutral'
    reason = (item.get('reasons') or ['Ограничений не выявлено'])[0]
    gap='нет ≥7 дней' if not gaps else '; '.join(f"{g['days']} дней" for g in gaps)
    st.markdown(f'<div class="nazar-health"><div class="nazar-health-title">{esc(region)}</div>'
                f'{status_badge(label,tone)}<div class="nazar-action-note" style="margin-top:10px">'
                f'Последняя дата: {esc(day(item["last_date"]))}<br>'
                f'Основная проблема: {esc(reason if item["status"] != "OK" else "нет")}'
                f'<span class="nazar-health-gap">Пропуски: {esc(gap)}</span></div></div>',unsafe_allow_html=True)
