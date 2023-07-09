# %%
# Linear Regression using  Scikit Learn 

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression


# %%
X , y = make_regression(n_samples=1000, n_features=2, n_informative=2, noise=10, random_state=1)

X.shape, y.shape

# %%
plt.subplot(1,2,1)
plt.scatter(X[:, 0], y)
plt.subplot(1,2,2)
plt.scatter(X[:, 1], y)
plt.show()

# %%

from mpl_toolkits import mplot3d

fig = plt.figure(figsize= (10, 7))
ax = plt.axes(projection = "3d")

ax.scatter3D(X[:, 0], X[:, 1], y , color = "green")

plt.title("3D scatter plot")
plt.show()

# %%



