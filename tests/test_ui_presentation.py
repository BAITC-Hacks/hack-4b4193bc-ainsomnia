"""Presentation invariants: escaped labels, unknown values, and visible gaps."""
import copy
import unittest
from src.ui.components import action_card, metric_card, status_badge
from src.ui.health import freshness_chart


class Capture:
    def __init__(self): self.output=[]
    def markdown(self,value,**kwargs): self.output.append(value)


class PresentationTest(unittest.TestCase):
    def test_action_escapes_labels_without_mutating_service_result(self):
        item={'severity':'HIGH','type':'NEW_SPIKE','region':'<script>alert(1)</script>',
              'topic':'A & B','headline':'new','data_as_of':'2025-01-01',
              'evidence':{'count':120,'baseline':10.,'excess':110.,'multiple':12.}}
        saved=copy.deepcopy(item);out=Capture();action_card(out,item)
        self.assertEqual(item,saved)
        self.assertIn('&lt;script&gt;',out.output[0])
        self.assertNotIn('<script>',out.output[0])
        self.assertIn('+110',out.output[0])
        self.assertIn('Высокий приоритет',out.output[0])
        self.assertNotIn('NEW_SPIKE',out.output[0])

    def test_unknown_metric_and_status_have_text(self):
        out=Capture();metric_card(out,'Очередь','—','Нет готовой модели')
        self.assertIn('Нет готовой модели',out.output[0])
        self.assertIn('Использовать нельзя',status_badge('Использовать нельзя','blocked'))

    def test_timeline_does_not_bridge_known_gap(self):
        health={'regions':{'Регион':{'first_date':'2020-01-01','last_date':'2020-01-31',
                    'gaps_ge7':[{'first_date':'2020-01-10','last_date':'2020-01-20','days':11}]}}}
        original=copy.deepcopy(health);fig=freshness_chart(health)
        self.assertEqual(health,original)
        self.assertEqual(len(fig.data),3)
        self.assertEqual(str(fig.data[0].x[-1])[:10],'2020-01-10')
        self.assertEqual(str(fig.data[1].x[0])[:10],'2020-01-10')
        self.assertEqual(str(fig.data[1].x[-1])[:10],'2020-01-21')
        self.assertEqual(fig.data[1].line.color,'#9A6500')
        self.assertEqual(str(fig.data[2].x[0])[:10],'2020-01-21')


if __name__=='__main__':unittest.main()
