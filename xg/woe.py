# import libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier as XGBC
from xgboost import DMatrix as DMat
from xgboost import cv as XGBCV
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.model_selection import train_test_split as t_t_s
from sklearn.metrics import accuracy_score as a_s
from sklearn.preprocessing import LabelEncoder as LE

print('\n all libs loaded...')

# define module variables
validation_size = 0.33
seed = 7

# set random seed
np.random.seed(seed)

# load CSV into pandas

csvfile = './woe-train.csv'
dataset = pd.read_csv(csvfile, header=0)

# delete 'status' and 'fil' col
dataset = dataset.drop(['fil', 'status'], axis=1)

# separate into X and Y
X = dataset.iloc[:, :17]
Y = dataset.iloc[:, 17]
X = X.values
Y = Y.values
print(Y.shape)
encoder = LE()
encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
Y = encoder.transform(Y)

X_train, X_test, Y_train, Y_test = t_t_s(X, Y, test_size=validation_size, random_state=seed, shuffle=False)

# instantiate XGBC class using defaults
model = XGBC()

# fit model to training datasets
print('\n training d model...')
model.fit(X_train, Y_train)

# view trained model
print('\n model...')
print(model)

# make predictions for test data
print('\n making predictions...')
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = a_s(Y_test, predictions)
print("WoE Init Accuracy: %.2f%%" % (accuracy * 100.0))

# load validation dataset
validationfile = './woe-test.csv'
dataset = pd.read_csv(validationfile, header=0)

dataset = dataset.drop(['fil', 'status_log'], axis=1)

Y_ext = dataset.filter(['status'], axis=1)
print('\n Y_ext.head(5)')
print(Y_ext.head(5))

print('\n Y_ext.values')
print(Y_ext.values)

print('\n Y_ext.shape')
print(Y_ext.shape)

#Y_ext = Y_ext.values

Y_rav = Y_ext.values.ravel()
print('\n Y_rav.shape')
print(Y_rav.shape)

encoder = LE()
encoder.fit(Y_rav)
Y_val = encoder.transform(Y_rav)

X_val = dataset.drop(['status'], axis=1)
X_val = X_val.values

# make predictions for validation data
print('\n making validation predictions...')
y_pred = model.predict(X_val)
val_predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = a_s(Y_val, val_predictions)
print("WoE validation Accuracy: %.2f%%" % (accuracy * 100.0))
