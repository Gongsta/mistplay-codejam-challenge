import pandas as pd
import numpy as np

dataset = pd.read_csv('./book2.csv')

dataset = dataset.fillna(0)

clean_dataset = dataset.loc[dataset['amount_spend']==0]
empty_dataset = dataset.loc[dataset['amount_spend']!=0]

empty_dataset = pd.concat([empty_dataset]*10)

dataset = pd.concat([clean_dataset, empty_dataset])

dataset = dataset.sample(frac=1)



X = dataset.iloc[:, [3,4,9,10,11,12,13,14,16,17,18,19]].values #3,4,9,10,11,12,
y = dataset.iloc[:, 15].values

newX = dataset.iloc[:, [3,4,9,10,11,12,13,14,16,17,18,19]].values #3,4,9,10,11,12,


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder = LabelEncoder()
X[:, 2] = X[:, 2].astype('str')
X[:, 2] = labelencoder.fit_transform(X[:, 2])
labelencoder2 = LabelEncoder()
X[:, 3] = labelencoder2.fit_transform(X[:, 3])
labelencoder3 = LabelEncoder()
X[:, 4] = X[:, 4].astype('str')
X[:, 4] = labelencoder3.fit_transform(X[:, 4])
labelencoder4 = LabelEncoder()
X[:, 5] = X[:, 5].astype('str')
X[:, 5] = labelencoder4.fit_transform(X[:, 5])
labelencoder5 = LabelEncoder()
X[:, 7] = X[:, 7].astype('str')
X[:, 7] = labelencoder5.fit_transform(X[:, 7])

'''
onehotencoder = OneHotEncoder(categorical_features = [2,3,4,5], handle_unknown='error', n_values='auto', sparse=True)
X = onehotencoder.fit(X)
X = onehotencoder.transform(X)

'''

'''
y = y.astype('str')
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

y = y.reshape(-1, 1)
'''


#Encoding y
import keras
y = y.astype('str')
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = keras.utils.to_categorical(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn import preprocessing
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X = sc.fit_transform(X)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pydot
import graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.utils import resample


model = Sequential()
model.add(Dense(15, input_dim=12, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image(filename='model_plot.png')

model.fit(X_train,y_train,epochs=100)
"""
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(15, input_dim=12, activation='relu'))
	model.add(Dense(5, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense  #used to build the hidden Layers


estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=10, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

"""
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)#0 = setosa 1 = versicolor 2 = virginica

labelencoder = LabelEncoder()
X[:, 2] = X[:, 2].astype('str')
X[:, 2] = labelencoder.fit_transform(X[:, 2])
labelencoder2 = LabelEncoder()
X[:, 3] = labelencoder2.fit_transform(X[:, 3])
labelencoder3 = LabelEncoder()
X[:, 4] = X[:, 4].astype('str')
X[:, 4] = labelencoder3.fit_transform(X[:, 4])
labelencoder4 = LabelEncoder()
X[:, 5] = X[:, 5].astype('str')
X[:, 5] = labelencoder4.fit_transform(X[:, 5])
labelencoder5 = LabelEncoder()
X[:, 7] = X[:, 7].astype('str')
X[:, 7] = labelencoder5.fit_transform(X[:, 7])

new_pred = np.array([[0.0, 0,0, 'country_0', '(40.0, 45.0]',0,0.0,0,14915.0, 9.0, 79.0, 3.0]])
new_pred[:, 2] = new_pred[:, 2].astype('str')
new_pred[:, 2] = labelencoder.transform(new_pred[:, 2])
new_pred[:, 3] = labelencoder2.transform(new_pred[:, 3])
new_pred[:, 4] = new_pred[:, 4].astype('str')
new_pred[:, 4] = labelencoder3.transform(new_pred[:, 4])
new_pred[:, 5] = new_pred[:, 5].astype('str')
new_pred[:, 5] = labelencoder4.transform(new_pred[:, 5])
new_pred[:, 7] = new_pred[:, 7].astype('str')
new_pred[:, 7] = labelencoder5.transform(new_pred[:, 7])


new_prediction = model.predict(sc.transform(new_pred))
new_prediction = np.argmax(new_prediction,axis=1)#0 = setosa 1 = versicolor 2 = virginica
