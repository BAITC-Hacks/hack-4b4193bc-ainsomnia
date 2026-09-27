"""Shift conversion, validation, display boundaries and calculator smoke checks."""
import tempfile
import unittest
from pathlib import Path

from src.capacity_scenario import HORIZON_WEEKS, shift_capacity, utilization_status


class CapacityScenarioTest(unittest.TestCase):
    def scenario(self, **overrides):
        args = dict(demand=3000., operators_per_shift=1, hours_per_shift=8., shifts_per_week=5.,
                    avg_handling_minutes=10., reserve_pct=0., forecast_weeks=13)
        args.update(overrides)
        return shift_capacity(**args)

    def test_manual_capacity_with_and_without_reserve(self):
        rows = self.scenario()
        self.assertEqual(rows[1]['raw_capacity'], 3120)
        self.assertEqual(rows[1]['available_capacity'], 3120)
        reserved = self.scenario(reserve_pct=20)
        self.assertEqual(reserved[1]['raw_capacity'], 3120)
        self.assertEqual(reserved[1]['available_capacity'], 2496)
        self.assertEqual(reserved[1]['balance'], -504)
        self.assertAlmostEqual(reserved[1]['utilization'], 3000/2496)

    def test_demand_scenarios_share_capacity_and_horizon(self):
        rows = self.scenario()
        self.assertEqual([r['scenario'] for r in rows], ['-20%', 'base', '+20%'])
        self.assertEqual([r['demand'] for r in rows], [2400, 3000, 3600])
        self.assertTrue(all(r['weeks'] == HORIZON_WEEKS == 13 for r in rows))
        self.assertTrue(all(r['available_capacity'] == 3120 for r in rows))
        with self.assertRaisesRegex(ValueError, '13 недель'):
            self.scenario(forecast_weeks=12)
        self.assertEqual(self.scenario(demand=0)[1]['utilization'], 0)

    def test_invalid_values_do_not_divide_by_zero(self):
        cases = [('operators_per_shift', 0), ('operators_per_shift', 1.5),
                 ('hours_per_shift', 0), ('hours_per_shift', 25),
                 ('shifts_per_week', 0), ('shifts_per_week', -1),
                 ('avg_handling_minutes', 0), ('avg_handling_minutes', -1),
                 ('reserve_pct', -1), ('reserve_pct', 95.01), ('demand', -1)]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.scenario(**{key: value})
        for key in ('demand', 'operators_per_shift', 'hours_per_shift', 'shifts_per_week',
                    'avg_handling_minutes', 'reserve_pct', 'forecast_weeks'):
            for value in (float('nan'), float('inf'), None, '8', True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.scenario(**{key: value})
        with self.assertRaises(ValueError):
            self.scenario(shifts_per_week=1e308)
        with self.assertRaises(ValueError):
            self.scenario(avg_handling_minutes=1e-320)
        self.assertGreater(self.scenario(reserve_pct=95, hours_per_shift=24, shifts_per_week=2.5)[1]['available_capacity'],0)

    def test_status_boundaries_use_unrounded_values(self):
        for value, expected in [(0.85,'Есть резерв'), (0.850001,'Высокая загрузка'),
                                (1.,'Высокая загрузка'), (1.000001,'Мощности недостаточно')]:
            with self.subTest(value=value):
                label, tone = utilization_status(value)
                self.assertEqual(label, expected)
                self.assertNotEqual(tone, 'blocked')

    def test_ui_blank_manual_invalid_and_missing_forecast(self):
        from streamlit.testing.v1 import AppTest
        from src.ui.capacity import DISCLAIMER, SCENARIO_NOTE
        script = '''import streamlit as st
from src.ui.capacity import capacity_section
forecast = [{'region':'Тестовый регион','topic':None,'weeks':13,'demand':3000.,
             'data_as_of':'2025-01-01','horizon_end':'2025-04-02'}]
forecast[0]['weeks'] = st.session_state.get('weeks', 13)
capacity_section(st, [] if st.session_state.get('missing') else forecast)
'''
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'calculator.py';path.write_text(script)
            at=AppTest.from_file(str(path),default_timeout=30).run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.number_input),5)
            self.assertTrue(all(v.value is None for v in at.number_input))
            text=lambda:'\n'.join(m.value for m in at.markdown)
            self.assertIn(DISCLAIMER,text());self.assertIn(SCENARIO_NOTE,text())
            self.assertEqual(text().count('<article class="nazar-capacity-card"'),3)
            for key,value in [('cap_operators',1),('cap_hours',8.),('cap_shifts',5.),('cap_aht',10.),('cap_reserve',20.)]:
                at.number_input(key=key).set_value(value)
            at.run();self.assertFalse(at.exception)
            self.assertEqual(text().count('2 496'),3)
            self.assertIn('−504',text());self.assertIn('120,2 %',text())
            at.number_input(key='cap_aht').set_value(0.).run()
            self.assertFalse(at.exception)
            self.assertTrue(any('больше нуля минут' in w.value for w in at.warning))
            self.assertNotIn('2 496',text())
            at.session_state['weeks']=12;at.run()
            self.assertFalse(at.exception)
            self.assertTrue(any('те же 13 недель' in w.value for w in at.warning))
            self.assertNotIn('<article class="nazar-capacity-card"', text())
            self.assertFalse(any('Одинаковый горизонт' in c.value for c in at.caption))
            at.session_state['missing']=True;at.run()
            self.assertFalse(at.exception)
            self.assertTrue(any('Нет доступного прогноза' in x.value for x in at.info))
            self.assertIn(DISCLAIMER,text())


if __name__ == '__main__': unittest.main()
