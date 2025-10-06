import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib

data = pd.read_csv('data/scoring.csv')
X = data.drop(columns=['default']).values
y = data['default'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#LogisticRegression Model
model_logreg = LogisticRegression(
    random_state=42,
    class_weight='balanced',
    max_iter=1000,
    C=0.1
)
model_logreg.fit(X_train, y_train)
y_pred_log = model_logreg.predict(X_test)
f1_log = f1_score(y_test, y_pred_log)
print(f"F1-score на тесте для ЛогРегрессии: {f1_log:.4f}\n")


#RandomForest Model
model_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5
)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf)
print(f"F1-score на тесте для Случайного Леса: {f1_rf:.4f}\n")


#GradientBoosting Model
sample_weights = compute_sample_weight('balanced', y_train)
model_gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=2,
    subsample=0.7,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
model_gb.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_gb = model_gb.predict(X_test)
f1_gb = f1_score(y_test, y_pred_gb)
print(f"F1-score на тесте для Градиентного Бустинга: {f1_gb:.4f}")


joblib.dump(model_gb, 'model.pkl')