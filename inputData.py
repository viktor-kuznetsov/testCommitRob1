import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('train_data.csv')
train_df = pd.DataFrame(train)
test = pd.read_csv('test_data.csv')
test_df = pd.DataFrame(test)

X_train = train_df.iloc[:,0:145]
y_train = train_df.iloc[:,145]
X_test = test_df.iloc[:,0:145]
y_test = test_df.iloc[:,145]

X_train_np = X_train.as_matrix()
y_train_np = y_train.as_matrix()

#print(X_train_np[:,0].shape)
print(y_train_np.shape)

fig = plt.figure()
plt.scatter(X_train_np[:,70], y_train_np, s=1)
plt.show()

#print(train_df[train_df.year > 2050])