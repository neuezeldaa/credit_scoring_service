import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score

import joblib

data = pd.read_csv('data/scoring.csv')
X = data.drop(columns=['default']).values
y = data['default'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#LogReg Model
model_logreg = LogisticRegression(class_weight='balanced')
model_logreg.fit(X_train, y_train)
y_pred = model_logreg.predict(X_test)
precision = precision_score(y_test, y_pred)
print(f"Отказано: {y_pred.mean() * 100:.0f}%")
print(f"Точность на линейной регрессии: {precision * 100:.0f}%\n")

#RandomForest Model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
precision = precision_score(y_test, y_pred)
print(f"Отказано: {y_pred.mean() * 100:.0f}%")
print(f"Точность на случайном лесе: {precision * 100:.0f}%\n")

#GradientBoostingModel
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                         max_depth=3, random_state=42)
model_gb.fit(X_train, y_train)
y_pred = model_gb.predict(X_test)
precision = precision_score(y_test, y_pred)
print(f"Отказано: {y_pred.mean() * 100:.0f}%")
print(f"Точность на градиентном бустинге: {precision * 100:.0f}%\n")




joblib.dump(model_logreg, 'model.pkl')