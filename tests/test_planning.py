import unittest
import pandas as pd
from src.planning import capacity, recurrence


class PlanningTest(unittest.TestCase):
    def test_capacity_units_and_bounds(self):
        rows = capacity(240, 2, 10, 5, .2)
        self.assertEqual([r['demand'] for r in rows], [192,240,288])
        self.assertEqual(rows[1]['available_capacity'],192)
        self.assertEqual(rows[1]['required_operator_hours'],25)
        self.assertIsNone(capacity(10,0,10,5,0)[1]['utilization'])
        for args in [(1,1,1,0,0), (1,1,1,1,1), (1,1.5,1,1,0), (float('nan'),1,1,1,0)]:
            with self.assertRaises(ValueError): capacity(*args)
        with self.assertRaises(ValueError): capacity(1,1,1,1,0,weeks=1)

    def test_censoring_boundary_gaps_and_empty(self):
        ev = pd.DataFrame({'регион':['R']*3,'тема':['T']*3,
                           'дата':pd.to_datetime(['2020-01-01','2020-01-08','2020-03-01'])})
        ev['конец']=ev['дата']
        h = {'regions':{'R':{'last_date':'2020-03-11','gaps_ge7':[]}}}
        row=recurrence(ev,h)[0]
        self.assertEqual(row['repeats_7d'],1)
        self.assertEqual(row['observable_7d'],3)
        self.assertEqual(row['observable_30d'],2)
        self.assertEqual(row['observable_60d'],2)
        self.assertEqual(row['censored_30d'],1/3)
        h['regions']['R']['gaps_ge7']=[{'first_date':'2020-01-03','last_date':'2020-01-09'}]
        row=recurrence(ev,h)[0]
        self.assertEqual(row['observable_7d'],1)
        self.assertEqual(recurrence(ev.iloc[:0],h),[])
        self.assertIsNone(recurrence(ev.iloc[-1:],h)[0]['recurrence_30d'])
        self.assertIsNone(recurrence(ev.iloc[-1:],h)[0]['median_days_to_next_observed'])


if __name__ == '__main__': unittest.main()
