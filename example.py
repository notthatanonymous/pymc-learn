from warnings import filterwarnings
filterwarnings("ignore")
import os

#os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['THEANO_FLAGS'] = 'device=cpu'    
    
import numpy as np
import pandas as pd
import pymc3 as pm

import pmlearn
from pmlearn.neural_network import MLPClassifier
print('Running on pymc-learn v{}'.format(pmlearn.__version__))

from sklearn import datasets
from sklearn.preprocessing import scale
import theano
floatX = theano.config.floatX

data = datasets.load_breast_cancer()

X, y = data.data, data.target

X = scale(X)
X = X.astype(floatX)
y = y.astype(floatX)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


model = MLPClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n\n\n")
print(f"Score: {model.score(X_test, y_test)}")
print("\n\n\n")
