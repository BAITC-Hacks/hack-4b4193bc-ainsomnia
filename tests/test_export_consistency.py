import io
import unittest
import pandas as pd
from src.export import summary_counts, export_frames, build_excel
from src.brief import render_html, header_lines
from src.operations import unavailable


class ConsistencyTest(unittest.TestCase):
    def test_same_scope_counts_in_export_and_brief(self):
        frame=pd.DataFrame({'region':['R','R','S','S'], 'appeal_class':['problem','info','system','problem'],
                            'topic':['T']*4,'created_at':pd.to_datetime(['2020-01-01']*4)})
        events=pd.DataFrame({'count':[7,10]})
        for selected in (frame,frame[frame.region=='R']):
            counts=summary_counts(selected)
            tables=export_frames(selected,events)
            self.assertEqual(counts['row_count'],len(selected))
            self.assertEqual(sum(counts['class_counts'].values()),len(selected))
            self.assertEqual(tables['Сводка по регионам']['всего'].sum(),len(selected))
            content=build_excel(selected,events,['R'],(pd.Timestamp('2020-01-01'),pd.Timestamp('2020-01-01')))
            ready=pd.read_excel(io.BytesIO(content),sheet_name='Сводка по регионам')
            self.assertEqual(int(ready['всего'].sum()),counts['row_count'])
            s=unavailable('test');s['summary']={**counts,'action_count':1,'spike_count':2}
            text=render_html(s,'fixed')
            self.assertIn(f"Всего строк: {len(selected)}",text)
            self.assertIn('всего действий: 1; всех всплесков: 2',text)
            # Both PDF and HTML call this same header contract.
            self.assertIn(f"Всего строк: {len(selected)}",header_lines(s,'fixed')[0])


if __name__=='__main__': unittest.main()
