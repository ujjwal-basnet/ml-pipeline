import numpy as np 
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier 
import os
import pickle

#fetch 
train_data =  pd.read_csv(os.path.join('data' , 'features' , 'train_bow.csv'))
x_train = train_data.iloc[: , 0:-1]
y_train = train_data.iloc[: , -1]

#define model and train
clf = GradientBoostingClassifier(n_estimators= 50)
clf.fit(x_train , y_train)

#save 
pickle.dump(clf,open('model.pkl' , 'wb'))
