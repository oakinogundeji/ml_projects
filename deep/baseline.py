# import libs
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier as KC
from keras.callbacks import ModelCheckpoint as MCP
from keras.models import model_from_json as m_f_j
from sklearn.preprocessing import LabelEncoder as LE
from sklearn import model_selection as m_s
from sklearn.model_selection import train_test_split as t_t_s

print('\n all libs loaded...')

validation_size = 0.33
seed = 7
np.random.seed(seed)

# load CSV into pandas

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

print('\ndataframes loaded...')

# check for presence of null values
'''print('\ntrain_dataset.isnull().sum()')
print(train_dataset.isnull().sum())
print('\ntest_dataset.isnull().sum()')
print(test_dataset.isnull().sum())'''

# we see one row of null values for train_dataset, this MUST be removed
print('\ndropping rows with NULL values')
print("\ntrain_dataset.dropna(axis=0, how='any', inplace=True)")
print(train_dataset.dropna(axis=0, how='any', inplace=True))

# from source notes, we know that missing data can appear as ' ?'
# this has to be removed
print("\n\nremoving values rep'd with ' ?'")
train_dataset_nomissing = train_dataset.replace(' ?', np.nan).dropna()
test_dataset_nomissing = test_dataset.replace(' ?', np.nan).dropna()

# from observation, we see that 'wage_class' value of train_dataset differs
# from test_datasetin that a '.' follows the class, this has to be fixed
# we will fix the issue on test_dataset
print("\nconverting incompatible values of 'wage_class' col in train and test datasets")
test_dataset_nomissing['wage_class'] = test_dataset_nomissing.wage_class.replace({' <=50K.': ' <=50K', ' >50K.':' >50K'})

# merge the datasets
print('\nmerging datasets...')
combined_set = pd.concat([train_dataset_nomissing, test_dataset_nomissing], axis = 0, ignore_index=True)

# view shape of combined dataset
print('\ncombined_set.shape...')
print(combined_set.shape)

# split the combined dataet into X and Y
print('\nsplitting combined set into X and Y')
X = combined_set.iloc[:, :14]
Y = combined_set.iloc[:, 14]

# apply one hot encoding against X
print('\napplying one hot encoding against X')
target_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
X = pd.get_dummies(X, columns=target_cols)
X = X.values

# apply label encoding against Y
print('\nlabel encoding Y...')
Y = Y.values
print('\nold Y.shape')
print(Y.shape)
encoder = LE()
encoder.fit(Y)
Y = encoder.transform(Y)

# split into train test sets using t_t_s
# because we combined the datasets to apply uniform
# one hot and label encoding, we set 'shuffle' parameter as false
# we also know that there should be 15060 rows in the test sets
print('\nsplitting X and Y into train and test datasets...')
test_set_size = test_dataset_nomissing.shape[0]
print('\ntest_set_size...')
print(test_set_size)
X_train, X_test, Y_train, Y_test = t_t_s(X, Y, test_size=test_set_size, random_state=seed, shuffle=False)

print('\n X_train.shape...')
print(X_train.shape)
print('\n Y_train.shape...')
print(Y_train.shape)
print('\n X_test.shape...')
print(X_test.shape)
print('\n Y_test.shape...')
print(Y_test.shape)

print('\n basic preprocessing finished...')

# create the model by doing the ff..
# 1. instantiate the Sequential class
# 2. add initial DEnse layer with 104 input neurons, 'normal'
# kernel_initializer, and 'relu' activation function
# 3. add dense output layer with 1 neuron, 'normal' kernel_initializer
# and 'sigmoid' activation function
# 4. compile the model using the 'binary_cross_entrophy' loss function
# 'adam' optimizer and setting 'accuracy' as the metric
# 5. return compiled model
model = Sequential()
model.add(Dense(104, input_dim=104, kernel_initializer='normal', activation='relu'))
model.add(Dense(104, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# define model checkpoint
filepath = "weights-best-baseline.hdf5"
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
with open("baseline-model.json", "w") as json_file:
    json_file.write(model_json)

# load json and create model
print('\n loading saved model from file...')
json_file = open('baseline-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = m_f_j(loaded_model_json)
# load weights into new model
print('\n loading saved best model weights from file...')
loaded_model.load_weights("weights-best-baseline.hdf5")
print("\n compiling model...")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('\n evaluating against test data set...')
# evaluate the model on test data
scores = loaded_model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
