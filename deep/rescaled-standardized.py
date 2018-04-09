# import libs
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier as KC
from keras.callbacks import ModelCheckpoint as MCP
from keras.models import model_from_json as m_f_j
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import StandardScaler as SS
from sklearn import model_selection as m_s
from sklearn.metrics import accuracy_score as a_s
from sklearn.metrics import confusion_matrix as c_m
from sklearn.metrics import cohen_kappa_score as c_k_s

print('\n all libs loaded...')

validation_size = 0.33
seed = 7
np.random.seed(seed)

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

# standardize X
scaler = SS().fit(X)
rescaledX = scaler.transform(X)

# split data into train and test datasets

X_train, X_test, Y_train, Y_test = m_s.train_test_split(rescaledX, Y, test_size=validation_size, random_state=seed)
print('\n basic preprocessing finished...')

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
model.add(Dense(108, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# define model checkpoint
filepath = "weights-best-rescaled-standardized.hdf5"
checkpoint = MCP(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# define parameters to be passed to KC
epochs = 500
batch_size = 50

# Fit the model
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=callbacks_list)

print('\n training ends...')
print('\n saving trained model to file...')
# serialize model to JSON
model_json = model.to_json()
with open("rescaled-standardized-model.json", "w") as json_file:
    json_file.write(model_json)

# load json and create model
print('\n loading saved model from file...')
json_file = open('rescaled-standardized-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = m_f_j(loaded_model_json)
# load weights into new model
print('\n loading saved best model weights from file...')
loaded_model.load_weights("weights-best-rescaled-standardized.hdf5")
print("\n compiling model...")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('\n evaluating against test data set...')
# evaluate the model on test data
scores = loaded_model.evaluate(X_test, Y_test, batch_size=batch_size)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

# make predictions
# predictions = loaded_model.predict(X_test, batch_size=batch_size)
