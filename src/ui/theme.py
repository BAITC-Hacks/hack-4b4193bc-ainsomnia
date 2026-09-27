"""Local visual tokens shared only by dashboard rendering."""
from pathlib import Path

NAVY = '#16324F'
TEAL = '#167D9A'
GREEN = '#39775E'
AMBER = '#9A6500'
RED = '#AB4040'
GREY = '#667085'
PALETTE = [NAVY, TEAL, '#5786AD', '#769C9A', '#827C9D', '#9A805D', GREY]


def apply_theme(st, presentation=False):
    css = Path(__file__).with_name('theme.css').read_text()
    if presentation:
        css += '''[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none}
        .nazar-metric-value {font-size:44px!important}
        .nazar-help, [data-testid="stExpander"]:has(.nazar-help-marker) {display:none}'''
    st.markdown('<style>'+css+'</style>', unsafe_allow_html=True)


def chart(fig):
    """Style a figure at the UI boundary. Export figures remain untouched."""
    fig.update_layout(template='plotly_white', font=dict(family='Arial, sans-serif',size=13,color=NAVY),
                      paper_bgcolor='white', plot_bgcolor='white', colorway=PALETTE,
                      margin=dict(l=16,r=24,t=64,b=48),
                      legend=dict(title_text='',orientation='h',y=-.2,x=0),
                      hoverlabel=dict(bgcolor='white',font_size=13),
                      modebar=dict(bgcolor='rgba(0,0,0,0)',color=GREY,activecolor=NAVY))
    for i,trace in enumerate(fig.data):
        if trace.type=='scatter' and str(trace.name).endswith('область'):
            trace.update(line_color=PALETTE[i % len(PALETTE)])
        if trace.type=='bar' and trace.name in ('Жалобы на городские проблемы','Справочные звонки','Служебные записи'):
            trace.update(marker_color={'Жалобы на городские проблемы':NAVY,'Справочные звонки':TEAL,'Служебные записи':GREY}[trace.name])
    fig.update_xaxes(gridcolor='#EDF1F5',zeroline=False,automargin=True)
    fig.update_yaxes(gridcolor='#EDF1F5',zeroline=False,automargin=True)
    return fig
