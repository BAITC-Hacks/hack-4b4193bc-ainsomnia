import unittest
import pandas as pd
from src.actions import queue, action


class ActionsTest(unittest.TestCase):
    def test_stability_and_block(self):
        args = ('NEW_SPIKE', 'HIGH', 'R', '2020-01-01', 'test')
        a = action(*args, {}, 'reason', 'check', 'spikes')
        b = action(*args, {'count': 99}, 'changed', 'check', 'spikes')
        self.assertEqual(a['stable_id'], b['stable_id'])
        self.assertEqual(queue({}, blocked='failed', events=object())[0]['severity'], 'BLOCKED')
        self.assertEqual(len(queue(None, events=object())), 1)

    def test_history_order_and_no_false_stale(self):
        h = {'regions': {'R': {'status': 'WARNING', 'rows': 10,
                              'last_date': '2020-01-31', 'reasons': ['gap']}}}
        ev = pd.DataFrame([{'регион': 'R', 'тема': 'T', 'дата': pd.Timestamp(d),
              'конец': pd.Timestamp(d), 'spike_type': typ, 'обращений': n,
              'медиана окна': 1., 'прирост': n-1, 'кратность': n}
              for d, n, typ in [('2020-01-01', 100, 'anomaly'), ('2020-01-30', 15, 'anomaly'),
                                ('2020-01-29', 20, 'anomaly'), ('2020-01-31', 30, 'seasonal')]])
        out = queue(h, events=ev, last_day=pd.Series({'R': pd.Timestamp('2020-01-31')}))
        self.assertEqual([x['evidence']['count'] for x in out if x['type']=='NEW_SPIKE'], [20,15,30])
        self.assertNotIn('DATA_STALE', [x['type'] for x in out])
        self.assertEqual(out[-1]['reason'], 'gap')
        self.assertEqual(out, queue(h, events=ev.iloc[::-1], last_day=pd.Series({'R': pd.Timestamp('2020-01-31')})))


if __name__ == '__main__':
    unittest.main()
