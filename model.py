# %% [markdown]
# ## Importing The Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Importing the dataset
# The dataset was collected from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

# %%
dataset = pd.read_csv("dataset/breast-cancer-wisconsin.csv", header=None)

# %% [markdown]
# ## Taking care of missing data
# Fill them up by the mean of the column

# %%
from sklearn.impute import SimpleImputer

# Replace '?' with None
imputer = SimpleImputer(missing_values="?", strategy='constant', fill_value=np.nan)
imputer = imputer.fit(dataset.iloc[:, 1:-1])
dataset.iloc[:, 1:-1] = imputer.transform(dataset.iloc[:, 1:-1])

# Replace None with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(dataset.iloc[:, 1:-1])
dataset.iloc[:, 1:-1] = imputer.transform(dataset.iloc[:, 1:-1])

# %% [markdown]
# ## Spliting the data into testset and training set

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[:, 1:-1].values, dataset.iloc[:, -1].values, test_size = 0.2)
print(x_train.shape)
print(x_test.shape)

# %% [markdown]
# ## Train the logistic regression model

# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# %% [markdown]
# ## Predicting test set result

# %%
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# %% [markdown]
# ## Confusion Matrix

# %%
# Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# %% [markdown]
# ## The accuracy with k-Fold Cross Validation

# %%
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


