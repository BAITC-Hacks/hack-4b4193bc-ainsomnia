"""Escaped presentation of existing aggregates; no scoring or detection."""
import html
from datetime import date

from src.ui.theme import NAVY, TEAL, GREEN, AMBER, RED, GREY

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
    st.markdown('<header class="nazar-header"><div><div class="nazar-brand">NAZAR-109</div>'
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
    section_header(st,'Что требует внимания','Сигналы для проверки, не подтверждённые аварии. Все регионы; последние доступные даты.')
    risk=snapshot['risk']; actions=snapshot['actions']
    warnings=sum(h['status']!='OK' for h in snapshot['health']['regions'].values())
    columns=st.columns(4)
    metric_card(columns[0],'Требует внимания',number(len(actions)),'Конкретных следующих проверок',NAVY)
    metric_card(columns[1],'Новых всплесков',number(snapshot['new_spikes']),'За последние 7 дней данных регионов',AMBER)
    metric_card(columns[2],'Очередь риска',number(risk.get('n')),'Только Караганда · исторический тест' if risk['available'] else 'Расчёт недоступен',TEAL)
    metric_card(columns[3],'Предупреждений качества',number(warnings),'Ограничения нужно учитывать',AMBER if warnings else GREEN)


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


def data_health_card(st, region, item):
    label,tone=('Данные готовы','ready') if item['status']=='OK' else ('Есть ограничения','attention')
    if item['status']=='BLOCKED':label,tone='Использовать нельзя','blocked'
    gaps=item['gaps_ge7']
    gap='нет ≥7 дней' if not gaps else '; '.join(f"{g['days']} дней" for g in gaps)
    st.markdown(f'<div class="nazar-health"><div class="nazar-health-title">{esc(region)}</div>'
                f'{status_badge(label,tone)}<div class="nazar-action-note" style="margin-top:10px">'
                f'Данные по {esc(day(item["last_date"]))}<br>Пропуски: {esc(gap)}</div></div>',unsafe_allow_html=True)
