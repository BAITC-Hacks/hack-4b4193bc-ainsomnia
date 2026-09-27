"""Additive explanations of the saved binary logistic model, no training."""
import numpy as np

LABELS = {'appeal_type': 'Тип обращения', 'region': 'Город', 'district': 'Район',
          'category': 'Категория службы', 'sub_category': 'Подкатегория', 'type': 'Тип записи',
          'hour': 'Час', 'dayofweek': 'День недели', 'month': 'Месяц', 'has_address': 'Наличие адреса'}


def explain(model, row, saved_probability=None):
    """Return only closed-list feature labels; never category values."""
    if len(row) != 1:
        raise ValueError('Exactly one row required')
    pre, clf = model.named_steps['pre'], model.named_steps['clf']
    if list(clf.classes_) != [0, 1] or clf.coef_.shape[0] != 1:
        raise ValueError('Binary classifier required')
    transformed = pre.transform(row)
    x = transformed.toarray()[0] if hasattr(transformed, 'toarray') else np.asarray(transformed)[0]
    contribution = x * clf.coef_[0]
    logit = float(clf.intercept_[0] + contribution.sum())
    probability = float(np.exp(-np.logaddexp(0, -logit)))
    expected = float(model.predict_proba(row)[0, 1])
    if not np.isclose(probability, expected, rtol=0, atol=1e-10):
        raise ValueError('Reconstruction mismatch')
    if saved_probability is not None and not np.isclose(probability, saved_probability, rtol=0, atol=5.1e-7):
        raise ValueError('Saved prediction mismatch')
    names = pre.get_feature_names_out()
    top = []
    for i in np.argsort(-contribution, kind='stable'):
        if contribution[i] <= 0 or len(top) == 3:
            break
        name = names[i].split('__', 1)[-1]
        key = next((k for k in sorted(LABELS, key=len, reverse=True)
                    if name == k or name.startswith(k + '_')), None)
        if key is None:
            raise ValueError('Unknown feature')
        top.append({'feature': LABELS[key], 'contribution_logit': float(contribution[i])})
    return {'available': True, 'intercept': float(clf.intercept_[0]),
            'sum_contributions': float(contribution.sum()), 'logit': logit,
            'probability': probability, 'reconstruction_error': abs(probability-expected), 'top': top}


def explain_rank(rank):
    from src import paths
    from src.risk_view import headline
    import pandas as pd
    try:
        import joblib
        headline()  # Enforces source agreement, including FAKE.
        rows = pd.read_csv(paths.PREDICTIONS).sort_values('y_prob', ascending=False, kind='mergesort')
        if not 0 <= rank < len(rows):
            raise ValueError('Rank outside queue')
        model = joblib.load(paths.MODEL)  # Only the locally generated trusted artifact.
        selected = rows.iloc[[rank]]
        return explain(model, selected, float(selected.y_prob.iloc[0]))
    except Exception:
        # Pickle/schema/version exceptions can contain raw dictionary values.
        return {'available': False, 'reason': 'Нет совместимой модели или объяснение не совпало с сохранённой оценкой'}
