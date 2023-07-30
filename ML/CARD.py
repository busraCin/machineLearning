import numpy as np
import pandas as pd
import pandas as ppd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
import graphviz
import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile


warnings.simplefilter(action='ignore', category=Warning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


df = pd.read_csv("datasets/diabetes.csv")
df.head(13)
df.shape
df.describe().T
df["Outcome"].value_counts()

#Modeling using CART
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
cart_model = DecisionTreeClassifier(random_state=1).fit(X,y)

# y_pred for Confusion matrix:
y_pred = cart_model.predict(X)

# y_pred for AUC:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)

# Model Validation: Holdout
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=45)
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

#Train Error
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
#accuracy 1.0
#f1-score 1.0
#out 1.0

#Test Error
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
#accuracy  0.69
#f1-score 0.54
#out 0.65

#Success Evaluation with CV
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.70
cv_results['test_f1'].mean()
# 0.57
cv_results['test_roc_auc'].mean()
# 0.67


#Hyperparameter Optimization with GridSearchCV
cart_model.get_params() 
cart_params = {'max_depth': range(1, 11), 
               "min_samples_split": range(2, 20)} 

cart_best_grid = GridSearchCV(cart_model,  
                              cart_params, 
                              cv=5, 
                              n_jobs=-1, 
                              verbose=1).fit(X, y) 

cart_best_grid.best_params_ #5-4 
cart_best_grid.best_score_ #0.75 
random = X.sample(1, random_state=45)
cart_best_grid.predict(random) 


#Final Model
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)
cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() #0.75
cv_results['test_f1'].mean() #0.61
cv_results['test_roc_auc'].mean() #0.79