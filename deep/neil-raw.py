# import libs
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier as KC
from keras.callbacks import ModelCheckpoint as MCP
from keras.models import model_from_json as m_f_j
from sklearn import model_selection as m_s
from sklearn.preprocessing import LabelEncoder as LE

print('all libs loaded...')

# load CSV into pandas

csvfile = './woe-train.csv'
dataset = pd.read_csv(csvfile, header=0)

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

# prepare to split into train n test datasets by defining d ff...
# 1. seed variable
# 2. validation_size variable spliting into 80/20 train/test

validation_size = 0.33
seed = 7
np.random.seed(seed)

# generate train test datasets
print('\n generating train test datasets...')
X_train, X_test, Y_train, Y_test = m_s.train_test_split(X, Y, test_size=validation_size, random_state=seed)

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
model.add(Dense(17, input_dim=17, kernel_initializer='normal', activation='relu'))
model.add(Dense(17, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# define model checkpoint
filepath = "weights-best-neil-raw.hdf5"
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
with open("neil-raw-model.json", "w") as json_file:
    json_file.write(model_json)

# load json and create model
print('\n loading saved model from file...')
json_file = open('neil-raw-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = m_f_j(loaded_model_json)
# load weights into new model
print('\n loading saved best model weights from file...')
loaded_model.load_weights("weights-best-neil-raw.hdf5")
print("\n compiling model...")

# load test dataset
testfile = './woe-test.csv'
dataset = pd.read_csv(testfile, header=0)

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

encoder = LE()
encoder.fit(Y_ext)
encoded_Y_ext = encoder.transform(Y_ext)

new_Y_ext = pd.DataFrame(encoded_Y_ext)

print('\n new_Y_ext.head(10)')
print(new_Y_ext.head(10))

# convert X and new_Y to numpy arrays
X_ext = X_ext.values
# convert Y into a 1D np ndarray
Y_ext = new_Y_ext.values
Y_ext = Y_ext.ravel()

print('\n Y_ext.shape')
print(Y_ext.shape)

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('\n evaluating against test data set...')
# evaluate the model on test data
scores = loaded_model.evaluate(X_ext, Y_ext)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
