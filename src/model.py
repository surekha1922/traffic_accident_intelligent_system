from preprocessing import load_and_preprocess

df = load_and_preprocess()

X = df.drop(columns=['Severity'])
y = df['Severity']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=20,
                               max_depth = 8,
                               n_jobs=-1,
                               random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


import pandas as pd

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print(feature_importances.head(10))


import matplotlib.pyplot as plt

feature_importances.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.show()