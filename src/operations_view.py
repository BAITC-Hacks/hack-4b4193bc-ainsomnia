"""Thin Streamlit rendering; numbers come from the operational services."""
from datetime import datetime, timezone
import pandas as pd
from src.ui.capacity import capacity_section
from src.closure import STATUS_RU
from src.brief import render_html, render_pdf
from src.export import PdfUnavailable
from src.ui.components import (action_card, section_header, info_callout, empty_state,
                               metric_card, number, esc, status_badge, day, spike_card)
from src.ui.theme import GREEN, AMBER, GREY


def action_section(st, snapshot):
    section_header(st, 'Требует внимания',
                   'Очередь всех регионов не зависит от фильтров аналитики. Приоритет означает порядок проверки, не аварийность.')
    def item(a, index):
        action_card(st, a)
        with st.expander(f"Открыть детали · {index + 1} · {a['region'] or 'Все регионы'}"):
            st.write(a['reason'])
            st.markdown('**Что проверить дальше**')
            st.write(a['recommended_next_check'])
            st.caption(f"Дата основания: {day(a['event_date'])}. Данные по: {day(a['data_as_of'])}.")
            if a['type'] == 'SLA_RISK':
                st.markdown('[Открыть очередь проверки](#risk)')
            elif a['type'] == 'NEW_SPIKE':
                st.markdown('[Открыть ленту и графики событий](#vspleski)')
    actions=snapshot['actions']
    visible = 3 if st.session_state.get('presentation') else 5
    for i,a in enumerate(actions[:visible]):
        item(a,i)
    if len(actions)>visible and not st.session_state.get('presentation'):
        with st.expander(f"Показать все {len(actions)}"):
            for i,a in enumerate(actions[visible:],start=visible):
                item(a,i)
    if not actions:
        empty_state(st,'Правила очереди не нашли действий в этом окне.')


def spikes_preview(st, snapshot):
    section_header(st, 'Новые всплески',
                   'Последние сигналы из очереди действий. Полная лента и графики находятся в разделе «Аналитика».')
    items = [item for item in snapshot['actions'] if item['type'] == 'NEW_SPIKE']
    if not items:
        empty_state(st, 'Новых всплесков в последних семи днях данных нет.')
        return
    shown = items[:3] if st.session_state.get('presentation') else items[:5]
    for start in range(0, len(shown), 3):
        columns = st.columns(min(3, len(shown) - start))
        for column, item in zip(columns, shown[start:start + 3]):
            spike_card(column, item)
    st.button('Подробнее о всплесках', key='open_analytics',
              on_click=lambda: st.session_state.update(page='analytics'))


def risk_preview(st, snapshot):
    section_header(st, 'Очередь проверки риска', 'Карагандинская область · исторический тест')
    risk = snapshot['risk']
    if not risk.get('available'):
        empty_state(st, 'Модель риска для текущего источника недоступна.')
        return
    from src.risk_view import headline, load_risk, top_share, WORK_SHARE
    try:
        metrics = headline()
        selected = top_share(load_risk(), WORK_SHARE)
    except (OSError, ValueError, KeyError):
        empty_state(st, 'Готовые результаты риска несовместимы с текущей сборкой.')
        return
    cols = st.columns(3)
    metric_card(cols[0], 'В очереди', number(selected['n']), 'обращений', AMBER)
    metric_card(cols[1], 'Целевых случаев', f"{selected['precision']:.0%}", 'в выбранной очереди', AMBER)
    metric_card(cols[2], 'Базовая доля', f"{metrics['base']:.0%}", 'во всём тестовом периоде', GREY)
    info_callout(st, 'Модель используется для приоритизации, а не как точная вероятность.')
    st.button('Открыть подробную очередь риска', key='open_risk',
              on_click=lambda: st.session_state.update(page='analytics'))


def planning_section(st, snapshot):
    st.subheader('Прогноз нагрузки')
    forecasts=snapshot['forecasts']
    if forecasts:
        chosen=st.selectbox('Регион и тема прогноза',range(len(forecasts)),key='forecast_view',
                            format_func=lambda i:forecasts[i]['region']+' · '+(forecasts[i]['topic'] or 'все темы'))
        f=forecasts[chosen]
        cols=st.columns(3)
        metric_card(cols[0],'Ожидаемая нагрузка',number(f['demand'],1),f"Жалоб за {f['weeks']} недель")
        metric_card(cols[1],'Горизонт до',day(f['horizon_end']),f"От данных по {day(f['data_as_of'])}")
        metric_card(cols[2],'Смена сезона','Да' if f['transition'] else 'Нет','В пределах выбранного горизонта',AMBER if f['transition'] else GREY)
        info_callout(st,'Прогноз оценивает нагрузку, не количество аварий. Он начинается от конца данных, не от сегодня. '
                     'Караганда — историческая проверка метода.')
        with st.expander('Метод, ограничения и все прогнозные ряды'):
            st.write(f['reason']);st.caption(f['limitation'])
            table=pd.DataFrame(forecasts).copy()
            table['topic']=table['topic'].fillna('Все темы')
            table['transition']=table['transition'].map({True:'Да',False:'Нет'})
            st.dataframe(table.rename(columns={
                'region':'Регион','topic':'Тема','model':'Модель','reason':'Почему',
                'data_as_of':'Последняя полная неделя','horizon_end':'Конец горизонта',
                'weeks':'Недель','demand':'Жалоб за горизонт','transition':'Смена сезона',
                'limitation':'Ограничение'}),hide_index=True,width='stretch')
    else:
        empty_state(st,'Прогноз недоступен: на тестовых данных рабочий прогноз не применяется, а на реальных требуется достаточная история.')
    capacity_section(st, forecasts)
    st.subheader('Повторное давление по теме')
    st.caption('Не повторная авария: нет безопасного адресного ключа. От конца одного всплеска до начала следующего '
               'той же темы. Неполное окно или пропуск ≥7 дней означает недостаточную наблюдаемость. Это не рейтинг регионов.')
    if snapshot['recurrence']:
        rows=pd.DataFrame(snapshot['recurrence'])
        region=st.selectbox('Регион повторного давления',sorted(rows.region.unique()))
        rows=rows[rows.region==region].drop(columns='region')
        names={'topic':'Тема','spikes':'Всплесков','data_as_of':'Данные по','observed_pairs':'Наблюдавшихся пар',
               'median_days_to_next_observed':'Условная медиана до следующего, дней'}
        for d in (7,30,60):
            names.update({f'observable_{d}d':f'Полных окон {d} д.',f'repeats_{d}d':f'Повторов {d} д.',
                          f'recurrence_{d}d':f'Доля повторов {d} д.',f'censored_{d}d':f'Доля ненаблюдаемых {d} д.'})
        compact=rows[['topic','spikes','repeats_30d','observable_30d','median_days_to_next_observed']].copy()
        compact['Наблюдаемость']=rows['censored_30d'].map(
            lambda x:'Недостаточно последующих данных для части событий' if x else 'Полные окна наблюдения')
        st.dataframe(compact.rename(columns=names),hide_index=True,width='stretch')
        with st.expander('Все интервалы повторного давления: 7, 30 и 60 дней'):
            st.dataframe(rows.rename(columns=names),hide_index=True,width='stretch')
        st.caption('Доли — от 0 до 1. Медиана только по наблюдавшимся парам без пропуска; не оценка с поправкой на цензурирование.')
    else:
        st.info('Нет событий для расчёта повторного давления.')


def closure_section(st, result):
    section_header(st,'Можно ли проверить факт решения?',
                   'Карта наблюдаемости проверок. Отсутствие данных не означает плохую работу службы.')
    if not result['available']:
        empty_state(st,result.get('reason','Нет проверенной карты'))
        return
    rows=pd.DataFrame(result['rows'])
    counts=rows.status.value_counts()
    columns=st.columns(3)
    for col,code,label,color in zip(columns,('VERIFIABLE','PARTIAL','NOT_OBSERVABLE'),
                                    ('Можно проверять','Частично','Данных недостаточно'),(GREEN,AMBER,GREY)):
        metric_card(col,label,number(counts.get(code,0)),'Проверок по регионам',color)
    tones={'VERIFIABLE':'ready','PARTIAL':'attention','NOT_OBSERVABLE':'neutral'}
    labels={'VERIFIABLE':'Можно проверять','PARTIAL':'Частично','NOT_OBSERVABLE':'Данных недостаточно'}
    channels=list(rows.channel.unique())
    matrix='<table class="nazar-matrix"><thead><tr><th>Регион</th>'+''.join('<th>'+esc(c)+'</th>' for c in channels)+'</tr></thead><tbody>'
    for region,group in rows.groupby('region',sort=False):
        matrix+='<tr><td>'+esc(region)+'</td>'
        for channel in channels:
            values=group[group.channel==channel]
            code=values.iloc[0]['status'] if len(values) else 'NOT_OBSERVABLE'
            matrix+='<td>'+status_badge(labels[code],tones[code])+'</td>'
        matrix+='</tr>'
    st.markdown(matrix+'</tbody></table>',unsafe_allow_html=True)
    info_callout(st,result.get('note','Карта не доказывает факт решения.'))
    with st.expander('Основания и ограничения проверок'):
        st.caption(result['snapshot_note'])
        rows['status']=rows.status.map(labels)
        for field in ('actual_close_available','open_visible','executor_available','sufficient_observations','noise_passed'):
            rows[field]=rows[field].map(lambda x:'да' if x is True else 'нет' if x is False else 'не определено / не проверялось')
        st.dataframe(rows.rename(columns={'region':'Регион','channel':'Канал','status':'Итог','reason':'Причина',
                      'actual_close_available':'Фактическая дата закрытия','open_visible':'Незакрытые видны',
                      'executor_available':'Безопасное поле исполнителя','sufficient_observations':'Достаточно наблюдений',
                      'noise_passed':'Проверки на шум пройдены','data_as_of':'Данные по'}),hide_index=True,width='stretch')
        st.caption('Акмола: «Передано в службу» считается незакрытым — допущение, требующее подтверждения заказчика.')


def brief_section(st, snapshot):
    section_header(st,'Оперативная сводка для руководителя',
                   'Сводка использует те же данные и правила, что текущая витрина. Все регионы; фильтры аналитики на сводку не влияют.')
    info_callout(st, 'Сигналы, их основания и следующие проверки — в одном документе. Выберите HTML или PDF после формирования.')
    if st.button('Сформировать оперативную сводку',type='primary',width='stretch'):
        stamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
        st.session_state['brief_html']=render_html(snapshot,stamp)
        st.session_state.pop('brief_pdf',None)
        try:
            st.session_state['brief_pdf']=render_pdf(snapshot,stamp)
        except PdfUnavailable as e:
            st.warning(str(e))
    if st.session_state.get('brief_html'):
        left,right=st.columns(2)
        left.download_button('HTML · оперативная сводка',st.session_state['brief_html'],file_name='nazar-brief.html',mime='text/html',width='stretch')
        if st.session_state.get('brief_pdf'):
            right.download_button('PDF · оперативная сводка',st.session_state['brief_pdf'],file_name='nazar-brief.pdf',mime='application/pdf',width='stretch')
