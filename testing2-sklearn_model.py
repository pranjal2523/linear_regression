# %%
# Linear Regression using  Scikit Learn 

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression


# %%
# Training Data

X1 = pd.read_csv('data/Linear_X_Train.csv')
Y1= pd.read_csv('data/Linear_Y_Train.csv')
X = X1.values
y = Y1.values

#Normalize
u = X.mean()
std = X.std()

X = (X-u)/std


# %%
# Test Data 
 
X_test = pd.read_csv("data/Linear_X_Test.csv").values

# %%
plt.scatter(X, y)


# %%

from sklearn.linear_model import LinearRegression

model = LinearRegression()


# %%
model.fit(X, y)

# %%
model.coef_

# %%
model.intercept_

# %%
X_test[1]

# %%
y_ = model.predict(X)

# %%
y_

# %%
model.score(X,y_)

# %%
plt.style.use('seaborn')
plt.scatter(X,y, label="Dataset") 
plt.plot(X, y_, color='red',label="Prediction")
plt.legend()
plt.show()

# %%



