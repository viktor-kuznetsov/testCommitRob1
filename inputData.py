import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split


load = pd.read_csv('train_data.csv')
df = pd.DataFrame(load)

X_df = df.iloc[:,0:145]
y_df = df.iloc[:,145]

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.25, random_state=64
)

print(X_train.shape)
print(X_test.shape)

cat_type = CategoricalDtype(categories = ["low", "moderate", "high"], ordered=True)
X_train["gamma_ray"] = X_train["gamma_ray"].astype(cat_type)
X_train["gamma_ray"] = X_train["gamma_ray"].cat.rename_categories([1,2,3])
X_test["gamma_ray"] = X_test["gamma_ray"].astype(cat_type)
X_test["gamma_ray"] = X_test["gamma_ray"].cat.rename_categories([1,2,3])


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

#for i in range(0, 145, 1):
    #print(type(X_train_np[0,i]))


lr = LinearRegression().fit(X_train, y_train)

print("Train Score: {}".format(lr.score(X_train, y_train)))
print("Test Score: {}".format(lr.score(X_test, y_test)))

#fig = plt.figure()
#plt.scatter(X_train_np[:,70], y_train_np, s=1)
#plt.show()
