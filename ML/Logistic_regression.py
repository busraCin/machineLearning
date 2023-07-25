"""
"Business Problem:
Developing a machine learning model capable of predicting whether individuals have diabetes when given certain features.
The dataset used for this task is part of a large dataset maintained at the National Institute of Diabetes and Digestive and Kidney Diseases in the United States.
It was collected for diabetes research on Pima Indian women residing in Phoenix, the 5th largest city in the state of Arizona, USA.
The data consists of 768 observations and 8 numerical independent variables.
The target variable is labeled as "outcome," indicating 1 for a positive diabetes test result and 0 for a negative test result."
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/diabetes.csv")
df.head()

#Target's analysis
df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()
#rate information of classes
100 * df["Outcome"].value_counts() / len(df)

#feature analysis
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

for col in df.columns:
    plot_numerical_col(df,col)

df.describe().T

# Target vs Features
cols = [col for col in df.columns if "Outcome" not in col]

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}),
          end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)


# Data Preprocessing
df.shape
df.head()
df.isnull().sum()
df.describe().T

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.5):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

# Model & Prediction
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
log_model = LogisticRegression().fit(X, y)
log_model.intercept_
log_model.coef_
y_pred = log_model.predict(X)
y_pred[0:10]
y[0:10]