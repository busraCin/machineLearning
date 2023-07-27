import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Exploratory Data Analysis
df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

# Data Preprocessing & Feature Engineering
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Modeling & Prediction
knn_model = KNeighborsClassifier().fit(X, y)
random_user = X.sample(1, random_state=45)
knn_model.predict(random_user)

# Model Evaluation

# y_pred for Confusion matrix:
y_pred = knn_model.predict(X) 

# y_pred for AUC:
y_prob = knn_model.predict_proba(X)[:, 1] 

print(classification_report(y, y_pred))
#precision 0.79
#recall 0.70
#f1-score 0.74
#accuacy 0.83

# AUC
roc_auc_score(y, y_prob)
# 0.90

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() #0.78
cv_results['test_f1'].mean() #0.59
cv_results['test_roc_auc'].mean() #0.78

knn_model.get_params()

# Hyperparameter Optimization