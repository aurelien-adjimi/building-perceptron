from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def selection_boruta(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42)
    boruta_selector.fit(X, y)
    return boruta_selector.support_, boruta_selector.support_weak_
