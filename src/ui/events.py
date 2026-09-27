"""Cards from event service rows; peak and starting baseline retain their meaning."""
from src.ui.components import metric_card, esc, number, day, status_badge


def event_cards(st, events):
    if events.empty:
        return
    columns=st.columns(min(3,len(events)))
    for i,(_,r) in enumerate(events.iterrows()):
        with columns[i % len(columns)]:
            st.markdown('<div class="nazar-health">'
                        f'{status_badge(r["тип"],"attention" if r["тип"]=="необычное" else "neutral")}'
                        f'<div class="nazar-action-region">{esc(r["регион"])}</div>'
                        f'<div class="nazar-action-note">{esc(r["тема"])}</div>'
                        f'<div class="nazar-action-value">+{number(r["прирост"],1)}</div>'
                        '<div class="nazar-action-note">жалоб сверх обычного</div>'
                        f'<div class="nazar-action-note">Пик {number(r["обращений"])} · обычно {number(r["медиана окна"],1)}'
                        f'<br>{number(r["кратность"],1)}× обычного · {number(r["дней подряд"])} дн.'
                        f'<br>Начало {day(r["дата"])} · пик {day(r["пик"])}</div></div>',unsafe_allow_html=True)
