import unittest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from src.risk_explanation import explain


class ExplanationTest(unittest.TestCase):
    def test_scaled_numeric_and_infrequent_categories(self):
        x = pd.DataFrame({'sub_category': ['a']*8+['b']*8+['rare1','rare2'], 'hour': range(18)})
        y = np.array([0,1]*9)
        model = Pipeline([('pre', ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=3), ['sub_category']),
            ('num', StandardScaler(), ['hour'])])), ('clf', LogisticRegression())]).fit(x,y)
        for i in (0,8,16,17):
            row = x.iloc[[i]]
            p = model.predict_proba(row)[0,1]
            e = explain(model, row, round(p,6))
            self.assertLessEqual(e['reconstruction_error'],1e-10)
            self.assertAlmostEqual(e['logit'], e['intercept']+e['sum_contributions'])
            self.assertTrue(all(v['contribution_logit']>0 for v in e['top']))
            self.assertLessEqual(len(e['top']),3)
        with self.assertRaises(ValueError):
            explain(model,x.iloc[[0]],0.999)


if __name__ == '__main__':
    unittest.main()
