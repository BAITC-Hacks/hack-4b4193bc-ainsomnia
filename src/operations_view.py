"""Thin Streamlit rendering; numbers come from the operational services."""
from datetime import datetime, timezone
import pandas as pd
from src.planning import capacity
from src.closure import STATUS_RU
from src.brief import render_html, render_pdf
from src.export import PdfUnavailable


def action_section(st, snapshot):
    st.subheader('Что требует внимания')
    st.caption('Все регионы; последние 7 дней данных каждого региона. Очередь и сводка не зависят '
               'от фильтров аналитики ниже. HIGH означает приоритет проверки, не аварийность.')
    rows=[]
    for a in snapshot['actions']:
        evidence = ', '.join(f'{k}: {v:.2f}' if isinstance(v,float) else f'{k}: {v}'
                             for k,v in a['evidence'].items())
        rows.append({'Приоритет':a['severity'],'Тип':a['type'],'Регион':a['region'],
                     'Тема':a['topic'],'Дата события':a['event_date'],'Данные по':a['data_as_of'],
                     'Что':a['headline'],'Факты':evidence,'Почему':a['reason'],
                     'Следующая проверка':a['recommended_next_check'],'Источник':a['source_module'],
                     'ID':a['stable_id']})
    if rows:
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch',height=300)
    else:
        st.info('Правила очереди не нашли действий в этом окне.')
    st.caption('count — пик жалоб; baseline — фон на старте; excess — жалоб сверх фона; '
               'multiple — кратность. n — размер очереди риска; precision — точность на историческом тесте.')
    if snapshot.get('health'):
        warnings=[r for r,h in snapshot['health']['regions'].items() if h['status']!='OK']
        if warnings:
            st.warning('Data Health: предупреждения — '+', '.join(warnings)+'. Причины в очереди и вкладке «Оперативно».')


def planning_section(st, snapshot):
    st.subheader('Прогноз нагрузки')
    forecasts=snapshot['forecasts']
    if forecasts:
        st.dataframe(pd.DataFrame(forecasts).rename(columns={
            'region':'Регион','topic':'Тема','model':'Модель','reason':'Почему',
            'data_as_of':'Последняя полная неделя','horizon_end':'Конец горизонта',
            'weeks':'Недель','demand':'Жалоб за горизонт','transition':'Смена сезона',
            'limitation':'Ограничение'}),hide_index=True,width='stretch')
        st.warning('Прогноз начинается от конца данных, не от сегодня. Караганда — историческая проверка метода. '
                   'Единица — жалоба problem, не все звонки.')
    else:
        st.info('Прогноз недоступен. Причины допуска регионов — в Data Health; на FAKE рабочий прогноз не применяется.')
    st.subheader('Сценарий нагрузки')
    st.caption('Сценарное изменение входящего потока −20% / base / +20% — не доверительный интервал. '
               'Параметры вводит пользователь: это не фактический штат 109 и не рекомендация найма.')
    if forecasts:
        index=st.selectbox('Ряд для сценария',range(len(forecasts)),format_func=lambda i:
                           forecasts[i]['region']+' · '+(forecasts[i]['topic'] or 'все темы'))
        selected=forecasts[index]
        st.write(f"Все часы ниже — на одного оператора ЗА ВСЕ {selected['weeks']} НЕДЕЛЬ после {selected['data_as_of']}.")
        c=st.columns(4)
        operators=c[0].number_input('Операторов',min_value=0,value=None,step=1)
        hours=c[1].number_input('Часов на оператора за 13 недель',min_value=0.,value=None)
        aht=c[2].number_input('Минут на одну жалобу (AHT)',min_value=0.01,value=None)
        reserve=c[3].number_input('Резерв, %',min_value=0.,max_value=99.99,value=None)
        if all(v is not None for v in (operators,hours,aht,reserve)):
            try:
                rows=capacity(selected['demand'],operators,hours,aht,reserve/100)
                st.dataframe(pd.DataFrame(rows).rename(columns={
                    'scenario':'Сценарий','weeks':'Недель','demand':'Спрос, жалоб',
                    'raw_capacity':'Мощность без резерва','available_capacity':'Доступная мощность',
                    'utilization':'Доля загрузки','balance':'Запас (+) / дефицит (−)',
                    'required_operator_hours':'Требуемые совокупные операторо-часы'}),hide_index=True)
                st.caption('Мощность = операторы × часы × 60 / AHT × (1 − резерв). При нулевой мощности загрузка не определена.')
            except ValueError as e:
                st.warning(str(e))
        else:
            st.info('Заполните четыре параметра; часы и AHT должны относиться к обработке жалоб выбранного ряда.')
    st.subheader('Повторное давление по теме')
    st.caption('Не повторная авария: нет безопасного адресного ключа. От конца одного всплеска до начала следующего '
               'той же темы. Неполное окно или пропуск ≥7 дней — not_observable, не false. Это не рейтинг регионов.')
    if snapshot['recurrence']:
        rows=pd.DataFrame(snapshot['recurrence'])
        region=st.selectbox('Регион повторного давления',sorted(rows.region.unique()))
        rows=rows[rows.region==region].drop(columns='region')
        names={'topic':'Тема','spikes':'Всплесков','data_as_of':'Данные по','observed_pairs':'Наблюдавшихся пар',
               'median_days_to_next_observed':'Условная медиана до следующего, дней'}
        for d in (7,30,60):
            names.update({f'observable_{d}d':f'Полных окон {d} д.',f'repeats_{d}d':f'Повторов {d} д.',
                          f'recurrence_{d}d':f'Доля повторов {d} д.',f'censored_{d}d':f'Доля ненаблюдаемых {d} д.'})
        st.dataframe(rows.rename(columns=names),hide_index=True,width='stretch')
        st.caption('Доли — от 0 до 1. Медиана только по наблюдавшимся парам без пропуска; не оценка с поправкой на цензурирование.')
    else:
        st.info('Нет событий для расчёта повторного давления.')


def closure_section(st, result):
    st.subheader('Можно ли проверить факт решения?')
    st.caption('Наблюдаемость конкретных каналов 5o; не рейтинг качества служб.')
    st.warning(result.get('note','Карта недоступна'))
    if not result['available']:
        st.info(result.get('reason','Нет проверенной карты'))
        return
    st.caption(result['snapshot_note'])
    rows=pd.DataFrame(result['rows'])
    rows['status']=rows.status.map(STATUS_RU)
    for field in ('actual_close_available','open_visible','executor_available','sufficient_observations','noise_passed'):
        rows[field]=rows[field].map(lambda x:'да' if x is True else 'нет' if x is False else 'не определено / не проверялось')
    st.dataframe(rows.rename(columns={'region':'Регион','channel':'Канал','status':'Итог','reason':'Причина',
                  'actual_close_available':'Фактическая дата закрытия','open_visible':'Незакрытые видны',
                  'executor_available':'Безопасное поле исполнителя','sufficient_observations':'Достаточно наблюдений',
                  'noise_passed':'Проверки на шум пройдены','data_as_of':'Данные по'}),hide_index=True,width='stretch')
    st.caption('Акмола: «Передано в службу» считается незакрытым — допущение, требующее подтверждения заказчика.')


def brief_section(st, snapshot):
    st.subheader('Оперативная сводка для руководителя')
    st.caption('Все регионы на момент данных, без LLM. Числа из того же snapshot, что в очереди и планировании.')
    if st.button('Подготовить HTML и PDF сводки'):
        stamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
        content=render_html(snapshot,stamp)
        st.download_button('Скачать HTML сводки',content,file_name='nazar-brief.html',mime='text/html')
        try:
            pdf=render_pdf(snapshot,stamp)
            st.download_button('Скачать PDF сводки',pdf,file_name='nazar-brief.pdf',mime='application/pdf')
        except PdfUnavailable as e:
            st.warning(str(e))
