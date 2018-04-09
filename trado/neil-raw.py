# import libs
import pandas
from pandas.tools.plotting import scatter_matrix as s_m
import matplotlib.pyplot as plt
from sklearn import model_selection as m_s
from sklearn.metrics import classification_report as c_r
from sklearn.metrics import confusion_matrix as c_m
from sklearn.metrics import accuracy_score as a_s
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB as GNB

print('all libs loaded...')

# load CSV into pandas

csvfile = './woe-train.csv'
dataset = pandas.read_csv(csvfile, header=0)

# summarize the dataset

# Dimensions: dataset.shape -> tells us how many rows and cols
# i.e. how many instances (rows) & attributes (cols)
print('\n dataset.shape:')
print(dataset.shape)

print('dataset.head(5)')
print(dataset.head(5))

# delete 'status' and 'fil' col
dataset = dataset.drop(['fil', 'status'], axis=1)

print('\n dataset.shape:')
print(dataset.shape)

print('dataset.head(5)')
print(dataset.head(5))

# get report on how many missing values exist in the entire dataset
print('\n dataset.isnull().sum()')
print(dataset.isnull().sum())

# separate into X and Y
X = dataset.iloc[:, :17]
Y = dataset.iloc[:, 17]

print('\n X.shape:')
print(X.shape)

# convert into numpy array
X = X.values
Y = Y.values

# prepare to split into train n test datasets by defining d ff...
# 1. seed variable
# 2. validation_size variable spliting into 80/20 train/test

seed = 7
validation_size = 0.2

# generate train test datasets
print('\n generating train test datasets...')
X_train, X_validation, Y_train, Y_validation = m_s.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# define 'scoring' parameter as 'accuracy'
scoring = 'accuracy'

# define array to hold candidate models
models = []

# instantiate candidate models and add to array
print('\n instantiating candidate models...')
models.append(('LR', LR()))
models.append(('LDA', LDA()))
models.append(('KNC', KNC()))
models.append(('DTC', DTC()))
models.append(('GNB', GNB()))

# run test harness
results = []
names = []
print('\n running test harness...')
for name, model in models:
    # 'kfold' var sets up the k-fold cross validation
    kfold = m_s.KFold(n_splits=10, random_state=seed)
    # 'cv_results' applies cross validation process to each model using the
    # training data i.e. features matrix X_train and results vector Y_train
    cv_results = m_s.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    # the data output is: name, mean & std
    model_res = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(model_res)

# LR performed best
print('\n')
print('running LR model i.e. best performing model on validation data....')
print('\n')
lr = LR()
# first fit the lr instance against the entire training dataset
lr.fit(X_train, Y_train)
# use the trained model to make predictions againts the validation
# feature matrix X_validation
predictions = lr.predict(X_validation)
# determine the accuracy of the model by scoring against the validation
# results vector Y_validation
print('accuracy_score:', a_s(Y_validation, predictions))
print('\n')
print('confusion_matrix')
print('\n')
# generate confusion matrix
print(c_m(Y_validation, predictions))
print('\n')
print('classification_report')
print('\n')
# generate classification report
print(c_r(Y_validation, predictions))

# load test dataset
testfile = './woe-test.csv'
dataset = pandas.read_csv(testfile, header=0)

# Dimensions: dataset.shape -> tells us how many rows and cols
# i.e. how many instances (rows) & attributes (cols)
print('\n dataset.shape:')
print(dataset.shape)

print('\ndataset.head(5)')
print(dataset.head(5))

# drop 'fil' and 'status_log' cols

dataset = dataset.drop(['fil', 'status_log'], axis=1)

# get report on how many missing values exist in the entire dataset
print('\n dataset.isnull().sum()')
print(dataset.isnull().sum())

print('\n dataset.shape:')
print(dataset.shape)

print('\n dataset.head(5)')
print(dataset.head(5))

Y_ext = dataset.filter(['status'], axis=1)
print('\n Y_ext.head(5)')
print(Y_ext.head(5))

X_ext = dataset.drop(['status'], axis=1)

print('\n X_ext.shape:')
print(X_ext.shape)

print('\n X_ext.head(5)')
print(X_ext.head(5))

# apply 'lr' against 'X_ext' and compare to 'Y_ext'

val_predictions = lr.predict(X_ext)
# determine the accuracy of the model by scoring against the validation
# results vector Y_validation
print('\n ext accuracy_score:', a_s(Y_ext, val_predictions))
print('\n')
print('ext confusion_matrix')
print('\n')
# generate confusion matrix
print(c_m(Y_ext, val_predictions))
print('\n')
print('ext classification_report')
print('\n')
# generate classification report
print(c_r(Y_ext, val_predictions))
