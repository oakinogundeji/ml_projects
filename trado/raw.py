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

csvfile = './adult.data.csv'
# the 'names' var below defines the colum names the dataframe will use
# since the csv file has no headers, we explicitly declare this
names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
         'occupation', 'relationship', 'race', 'sex', 'capital_gain',
         'capital_loss', 'hours_per_week', 'native_country', 'result']
dataset = pandas.read_csv(csvfile, header=None, names=names)

# summarize the dataset

# Dimensions: dataset.shape -> tells us how many rows and cols
# i.e. how many instances (rows) & attributes (cols)
print('\n dataset.shape:')
print(dataset.shape)

# Peek @ data
# 1st 10 rows: dataset.head(10)
#print('\n dataset.head(10)')
#print(dataset.head(10))

# last 10 rows: dataset.tail(10)
#print('\n dataset.tail(10)')
#print(dataset.tail(10))

# Statistical Summary: dataset.describe()
print('\n dataset.describe()')
print(dataset.describe())

# Data visualization
# Univariate plots of each attribute
#dataset.plot(kind='box',
#             subplots=True, layout=(3, 2), sharex=False, sharey=False)
#plt.show()
# histogram
#dataset.hist()
#plt.show()

# check if any data is msising in 1st 10 rows
#print('\n dataset.isnull().head(10)')
#print(dataset.isnull().head(10))

# check if any data is msising in last 10 rows
#print('\n dataset.isnull().tail(10)')
#print(dataset.isnull().tail(10))

# get report on how many missing values exist in the entire dataset
print('\n dataset.isnull().sum()')
print(dataset.isnull().sum())

# from the above, we see that only the last row has missing data
# so we can safely delete that row

# drop rows with missing data
print("\n dataset.dropna(axis=0, how='any', inplace=True)")
print(dataset.dropna(axis=0, how='any', inplace=True))

# check if any data is msising in last 10 rows
#print('\n dataset.isnull().tail(10)')
#print(dataset.isnull().tail(10))

# get report on how many missing values exist in the entire dataset
print('\n dataset.isnull().sum()')
print(dataset.isnull().sum())

print('\n dataset.shape:')
print(dataset.shape)

print(dataset.iloc[14809])

# perform one hot encoding on categorical data
# perform one hot encoding on categorical data
target_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
print('\n dataset = pandas.get_dummies(dataset)')
#pandas.get_dummies(dataset, columns=target_cols)
#dataset = pandas.get_dummies(dataset, columns=target_cols)
X = dataset.iloc[:, :14]
Y = dataset.iloc[:, 14]

print('\n X.shape:')
print(X.shape)

X = pandas.get_dummies(X, columns=target_cols)
print('\n X.shape: after 1 hot')
print(X.shape)
print('\n')
print(X.iloc[14809])
print(X.head(10))

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

# LDA performed best
print('\n')
print('running LDA model i.e. best performing model on validation data....')
print('\n')
lda = LDA()
# first fit the lda instance against the entire training dataset
lda.fit(X_train, Y_train)
# use the trained model to make predictions againts the validation
# feature matrix X_validation
predictions = lda.predict(X_validation)
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
