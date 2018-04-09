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
from sklearn.preprocessing import MinMaxScaler as MMS

print('\n all libs loaded...')

# define module variables
validation_size = 0.33
seed = 7

# set random seed
np.random.seed(seed)

# load CSV into pandas

trainfile = './adult.data.csv'
testfile = './adult.test.csv'
# the 'names' var below defines the colum names the dataframe will use
# since the csv file has no headers, we explicitly declare this
names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
         'occupation', 'relationship', 'race', 'sex', 'capital_gain',
         'capital_loss', 'hours_per_week', 'native_country', 'wage_class']
train_dataset = pd.read_csv(trainfile, header=None, names=names)
test_dataset = pd.read_csv(testfile, header=None, names=names)

print('\n dataframes loaded...')

# explore dataframes
'''print('\n train_dataset.info()')
print(train_dataset.info())
print('\n test_dataset.info()')
print(test_dataset.info())
print('\n train_dataset.head()')
print(train_dataset.head())
print('\n test_dataset.head()')
print(test_dataset.head())
print('\n train_dataset.tail()')
print(train_dataset.tail())
print('\n test_dataset.tail()')
print(test_dataset.tail())'''

# check for presence of null values
'''print('\n train_dataset.isnull().sum()')
print(train_dataset.isnull().sum())
print('\n test_dataset.isnull().sum()')
print(test_dataset.isnull().sum())'''

# we see one row of null values for train_dataset, this MUST be removed
print("\n train_dataset.dropna(axis=0, how='any', inplace=True)")
print(train_dataset.dropna(axis=0, how='any', inplace=True))

# get initial dimensions of train n test datasets
'''print('\n train_dataset.shape')
print(train_dataset.shape)
print('\n test_dataset.shape')
print(test_dataset.shape)'''

# from source notes, we know that missing data can appear as ' ?'
# this has to be removed
print("\n train_dataset.replace(' ?', np.nan).dropna().shape")
print(train_dataset.replace(' ?', np.nan).dropna().shape)
print("\n test_dataset.replace(' ?', np.nan).dropna().shape")
print(test_dataset.replace(' ?', np.nan).dropna().shape)

# execute removal of unkwon data
train_dataset_nomissing = train_dataset.replace(' ?', np.nan).dropna()
test_dataset_nomissing = test_dataset.replace(' ?', np.nan).dropna()

# get new dimensions of train n test datasets
print('\n train_dataset_nomissing.shape')
print(train_dataset_nomissing.shape)
print('\n test_dataset_nomissing.shape')
print(test_dataset_nomissing.shape)

# from observation, we see that 'wage_class' value of train_dataset differs
# from test_datasetin that a '.' follows the class, this has to be fixed
# we will fix the issue on test_dataset
test_dataset_nomissing['wage_class'] = test_dataset_nomissing.wage_class.replace({' <=50K.': ' <=50K', ' >50K.':' >50K'})

# view the results
'''print('\n test_dataset_nomissing.head()')
print(test_dataset_nomissing.head())
print('\n test_dataset_nomissing.tail()')
print(test_dataset_nomissing.tail())'''

# we want to convert categorical data to one hot encoded data
# we begin by merging the datasets
combined_set = pd.concat([train_dataset_nomissing, test_dataset_nomissing], axis = 0, ignore_index=True)

'''print('\n combined_set.head()')
print(combined_set.head())
print('\n combined_set.tail()')
print(combined_set.tail())
print('\n combined_set.info()')
print(combined_set.info())'''

# split the combined dataet into X and Y
X = combined_set.iloc[:, :14]
Y = combined_set.iloc[:, 14]

'''print('\n X.shape')
print(X.shape)
print('\n Y.shape')
print(Y.shape)'''

# apply one hot encoding against X
target_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
X = pd.get_dummies(X, columns=target_cols)
# apply label encoding against Y
Y = Y.values
print('\n old Y.shape')
print(Y.shape)
encoder = LE()
encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
Y = encoder.transform(Y)

#new_Y = pd.DataFrame(encoded_Y)

'''print('\n X.head(10): after 1 hot')
print(X.head(10))
print('\n new_Y.head(10)')
print(new_Y.head(10))'''

# convert X and new_Y to numpy arrays
X = X.values
print('\n new Y.shape')
print(Y.shape)

# minmax X
scaler = MMS(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# split into train test sets using t_t_s
# because we combined the datasets to apply uniform
# one hot and label encoding, we set 'shuffle' parameter as false
# we also know that there should be 15060 rows in the test sets
test_set_size = test_dataset_nomissing.shape[0]
print('\n test_set_size...')
print(test_set_size)
X_train, X_test, Y_train, Y_test = t_t_s(rescaledX, Y, test_size=test_set_size, random_state=seed, shuffle=False)

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
train_predictions = model.score(X_train, Y_train)

# determine the accuracy of the model by scoring against the test
# results vector Y_test
print('\n xgb_test_minmax_accuracy_score:', a_s(Y_test, predictions))
print('\n xgb_train_minmax_accuracy_score:', train_predictions)
