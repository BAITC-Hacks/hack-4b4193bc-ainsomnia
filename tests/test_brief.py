import unittest
from unittest.mock import patch
from src.actions import action
from src.brief import render_html, sections
from src.operations import snapshot, unavailable


class BriefTest(unittest.TestCase):
    def test_determinism_escaping_and_shared_numbers(self):
        s=unavailable('source mismatch')
        s.update(available=True,source='fake',new_spikes=7)
        s['actions']=[action('NEW_SPIKE','HIGH','R','2020-01-01','<script>',{'count':17},'reason','check','spikes')]
        s['risk']={'available':True,'n':20,'share':.2,'precision':.75,'cutoff':'2020-01-01','data_as_of':'2020-02-01'}
        out=render_html(s,'2026-01-01T00:00:00Z')
        self.assertEqual(out,render_html(s,'2026-01-01T00:00:00Z'))
        self.assertIn('&lt;script&gt;',out)
        self.assertNotIn('<script>',out)
        self.assertIn('ПОДДЕЛЬНЫЕ ДАННЫЕ',out)
        self.assertIn('регионов: 7',out)
        self.assertIn('20 обращений',out)
        self.assertEqual(len(sections(s)),6)

    def test_blocked_does_not_read_analytics(self):
        with patch('src.operations.load',return_value=(None,'failed build')), patch('src.operations.load_events') as events:
            s=snapshot()
            events.assert_not_called()
            self.assertFalse(s['available'])
            self.assertEqual(s['actions'][0]['severity'],'BLOCKED')
            self.assertIn('Недоступно',render_html(s,'now'))


if __name__=='__main__': unittest.main()
