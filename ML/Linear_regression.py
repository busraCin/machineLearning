import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("datasets/advertising.csv")
df.shape
df.head(10)
df.describe().T

#model
X = df[["TV"]]
y = df[["sales"]]
reg_model = LinearRegression().fit(X,y)

reg_model.intercept_[0] #bias b
reg_model.coef_[0][0] #weight w1

#How many sales would be expected if 150 units were spent on TV?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

#visualize the model
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Number of Sales")
g.set_xlabel("TV Spending")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

# Prediction Success
y.mean()
y.std()

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 10.51

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE
reg_model.score(X, y)
#0.61


#MULTİPLE LİNEAR REGRESSION
df= pd.read_csv("datasets/advertising.csv")
X = df.drop('sales', axis=1)
y = df[["sales"]]

#Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
y_test.shape
y_train.shape
reg_model2 = LinearRegression().fit(X_train, y_train)

reg_model2.intercept_ #b
reg_model2.coef_ #w1

# TV: 30
# radio: 10
# newspaper: 40
# 2.90
# 0.0468431 , 0.17854434, 0.00258619
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002 ,model equation
2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619
new_data = [[30],[10],[40]]
new_data = pd.DataFrame(new_data).T
reg_model2.predict(new_data)

#success prediction
# Train RMSE
y_pred = reg_model2.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN RKARE
reg_model2.score(X_train, y_train)

# Test RMSE
y_pred = reg_model2.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test RKARE
reg_model2.score(X_test, y_test)
#0.89

# 10-Fold  CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69


# 5-Fold CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71



# Simple Linear Regression with Gradient Descent from Scratch

# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

# update_weights
def update_weights(Y,b,w,X,learnin_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0,m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]

    new_b = b - (learnin_rate * 1/m * b_deriv_sum)
    new_w = w - (learnin_rate * 1/m * w_deriv_sum)
    return new_b, new_w

#train function
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

df = pd.read_csv("datasets/advertising.csv")
X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
