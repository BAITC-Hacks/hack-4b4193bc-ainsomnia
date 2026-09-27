"""Deterministic text from a service snapshot, rendered to HTML or PDF."""
import html
import io
import json

TITLE = 'NAZAR-109 — оперативная сводка'


def sections(snapshot):
    items = snapshot['actions']
    def item_line(a):
        return (f"{a['severity']} · {a['region'] or 'все регионы'} · {a['topic'] or 'все темы'}: "
                f"{a['headline']}. {json.dumps(a['evidence'],ensure_ascii=False,sort_keys=True)}. "
                f"{a['reason']}. Проверить: {a['recommended_next_check']}")
    health = snapshot.get('health')
    quality = ([f"{reg}: {h['status']}; {'; '.join(h['reasons']) or 'нет предупреждений'}"
                for reg,h in sorted(health['regions'].items())] if health else ['Профиль недоступен'])
    risk = snapshot['risk']
    risk_lines = ([f"Караганда: {risk['n']} обращений ({risk['share']:.0%} потока); "
                   f"precision {risk['precision']:.4f}, исторический тест с {risk['cutoff']} "
                   f"по {risk['data_as_of']}"] if risk['available'] else [risk['reason']])
    forecasts = [f"{f['region']} · {f['topic'] or 'все темы'}: {f['demand']:.0f} жалоб "
                 f"за {f['weeks']} недель после {f['data_as_of']}; {f['model']}. "
                 f"{f['reason']}. {f['limitation']}" for f in snapshot['forecasts']]
    return [
        ('1. Требует внимания — первые 10', [item_line(a) for a in items[:10]] or ['Нет действий']),
        ('2. Новые всплески', [f"За последние 7 дней данных своих регионов: {snapshot['new_spikes']}. "
                              'Это статистические события, не доказательство аварий.'] if snapshot['available']
                             else ['Недоступно: проверенная аналитика не прочитана']),
        ('3. Качество и свежесть данных', quality),
        ('4. Risk queue Караганда', risk_lines),
        ('5. Изменение ожидаемой нагрузки', forecasts or ['Прогноз для этой версии недоступен']),
        ('6. Что нельзя утверждать по этим данным', [
            'Окна регионов различаются; даты относятся к выгрузке, не к сегодня. Срок обновления не согласован.',
            'Риск — историческая очередь Караганды, не калиброванная вероятность и не прогноз всего Казахстана.',
            'Прогноз считает жалобы, не все контакты. Изменение состава потока ограничивает сравнения.',
            'Повтор по теме не доказывает повторную аварию; карта закрытий не является рейтингом служб.',
            'Ноль сигналов триангуляции не означает, что все работают хорошо.'])]


def header_lines(snapshot, generated_at):
    health = snapshot.get('health')
    totals=snapshot.get('summary')
    summary=([f"Всего строк: {totals['row_count']}; классы: "
              f"{json.dumps(totals['class_counts'],ensure_ascii=False,sort_keys=True)}; "
              f"всего действий: {totals['action_count']}; всех всплесков: {totals['spike_count']}."] if totals else [])
    return summary + ([f'Создано UTC: {generated_at}', 'ПОДДЕЛЬНЫЕ ДАННЫЕ — числа не описывают ни один регион.']
            if snapshot['source']=='fake' else [f'Создано UTC: {generated_at}', 'Источник: REAL']) + [
        'DATA AS OF: '+('; '.join(f"{reg}: {h['last_date']}" for reg,h in sorted(health['regions'].items()))
                        if health else 'данные недоступны')]


def render_html(snapshot, generated_at):
    esc = html.escape
    out = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
           f'<title>{esc(TITLE)}</title><body><h1>{esc(TITLE)}</h1>']
    out.extend(f'<p>{esc(line)}</p>' for line in header_lines(snapshot,generated_at))
    for title, lines in sections(snapshot):
        out.append(f'<h2>{esc(title)}</h2><ul>')
        out.extend(f'<li>{esc(line)}</li>' for line in lines)
        out.append('</ul>')
    return '\n'.join(out+['</body></html>'])


def render_pdf(snapshot, generated_at):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle
    from src.export import _register_font
    fonts = _register_font()
    normal = ParagraphStyle('brief',fontName=fonts['font'],fontSize=9,leading=13)
    heading = ParagraphStyle('heading',fontName=fonts['bold'],fontSize=13,leading=17,spaceAfter=8)
    story = [Paragraph(html.escape(TITLE),heading)]
    for line in header_lines(snapshot,generated_at):
        story.append(Paragraph(html.escape(line),normal))
    for title, lines in sections(snapshot):
        story.extend([Spacer(1,12),Paragraph(html.escape(title),heading)])
        for line in lines:
            story.extend([Paragraph(html.escape(line),normal),Spacer(1,5)])
    output=io.BytesIO()
    SimpleDocTemplate(output,invariant=1,title=TITLE,author='NAZAR-109').build(story)
    return output.getvalue()
