# import libs
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier as KC
from sklearn.model_selection import cross_val_score as c_v_s
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn import model_selection as m_s

print('\n all libs loaded...')

# load CSV into pandas

csvfile = './adult.data.csv'
# the 'names' var below defines the colum names the dataframe will use
# since the csv file has no headers, we explicitly declare this
names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
         'occupation', 'relationship', 'race', 'sex', 'capital_gain',
         'capital_loss', 'hours_per_week', 'native_country', 'result']
dataset = pd.read_csv(csvfile, header=None, names=names)

print('\n dataframe loaded...')

# Dimensions: dataset.shape -> tells us how many rows and cols
# i.e. how many instances (rows) & attributes (cols)
print('\n dataset.shape:')
print(dataset.shape)

# get report on how many missing values exist in the entire dataset
print('\n dataset.isnull().sum()')
print(dataset.isnull().sum())

# drop rows with missing data
print("\n dataset.dropna(axis=0, how='any', inplace=True)")
print(dataset.dropna(axis=0, how='any', inplace=True))

print('\n dataset.shape:')
print(dataset.shape)

# Split into X and Y

X = dataset.iloc[:, :14]
Y = dataset.iloc[:, 14]

print('\n X.shape:')
print(X.shape)

# perform one hot encoding on categorical data of X
target_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
print('\n dataset = pandas.get_dummies(dataset)')

X = pd.get_dummies(X, columns=target_cols)
print('\n X.shape: after 1 hot')
print(X.shape)

# perform categorical encoding of Y since d vals are a binary set
print('\n Y.head(10)')
print(Y.head(10))

encoder = LE()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

new_Y = pd.DataFrame(encoded_Y)

print('\n new_Y.head(10)')
print(new_Y.head(10))

# convert X and new_Y to numpy arrays
X = X.values
Y = new_Y.values

print('\n basic preprocessing finished...')

def create_model():
    # create the model by doing the ff..
    # 1. instantiate the Sequential class
    # 2. add initial DEnse layer with 108 input neurons, 'normal'
    # kernel_initializer, and 'relu' activation function
    # 3. add dense output layer with 1 neuron, 'normal' kernel_initializer
    # and 'sigmoid' activation function
    # 4. compile the model using the 'binary_cross_entrophy' loss function
    # 'adam' optimizer and setting 'accuracy' as the metric
    # 5. return compiled model
    model = Sequential()
    model.add(Dense(108, input_dim=108, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define parameters to be passed to KC
epochs = 100
batch_size = 100
verbose = 5

# define parameters to be passed to SKF
n_splits = 10
seed = 7
np.random.seed(seed)
scoring = 'accuracy'

# train model
estimator = KC(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=verbose)
kfold = m_s.KFold(n_splits=10, random_state=seed)
cv_results = m_s.cross_val_score(estimator, X, Y, cv=kfold, scoring=scoring)
model_res = "Raw: %f (%f)" % (cv_results.mean(), cv_results.std())
print(model_res)

# model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
