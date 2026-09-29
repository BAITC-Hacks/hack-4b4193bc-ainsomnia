"""Health aggregate cards and observation windows; no new quality thresholds."""
import pandas as pd
import plotly.graph_objects as go
from src.ui.components import data_health_card, section_header, info_callout
from src.ui.theme import chart, TEAL, GREY, AMBER


def health_overview(st, health, compact=False):
    section_header(st,'Качество и свежесть данных',
                   'Готовность данных не является оценкой работы региона.',anchor='health-summary' if compact else None)
    if not health:
        st.info('Профиль качества пока недоступен.')
        return
    def cards():
        rows=list(health['regions'].items())
        for start in range(0,len(rows),4):
            cols=st.columns(4)
            for col,(region,item) in zip(cols,rows[start:start+4]):
                data_health_card(col,region,item)
    if compact:
        warning=sum(h['status']!='OK' for h in health['regions'].values())
        info_callout(st,f"Регионов с ограничениями: {warning} из {len(health['regions'])}. "
                     'Проверьте даты и полноту перед решением.')
        cards()
        st.button('Подробнее о качестве данных', key='open_control',
                  on_click=lambda: st.session_state.update(page='control'))
    else:
        cards()


def freshness_chart(health):
    fig=go.Figure()
    for region,h in health['regions'].items():
        start=pd.Timestamp(h['first_date']);end=pd.Timestamp(h['last_date'])+pd.Timedelta(days=1)
        cursor=start
        for gap in h['gaps_ge7']:
            stop=pd.Timestamp(gap['first_date'])
            if cursor < stop:
                fig.add_scatter(x=[cursor,stop],y=[region,region],mode='lines',line=dict(color=TEAL,width=12),showlegend=False,
                                hovertemplate='%{y}<br>%{x|%d.%m.%Y}<extra></extra>')
            gap_end=pd.Timestamp(gap['last_date'])+pd.Timedelta(days=1)
            fig.add_scatter(x=[stop,gap_end],y=[region,region],mode='lines',line=dict(color=AMBER,width=5,dash='dot'),showlegend=False,
                            hovertemplate='%{y}<br>Разрыв данных: %{x|%d.%m.%Y}<extra></extra>')
            cursor=gap_end
        if cursor<end:
            fig.add_scatter(x=[cursor,end],y=[region,region],mode='lines',line=dict(color=TEAL,width=12),showlegend=False,
                            hovertemplate='%{y}<br>%{x|%d.%m.%Y}<extra></extra>')
    fig.update_layout(height=360,xaxis_title='Период наблюдений',yaxis_title=None)
    return chart(fig)
