import numpy as np 
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier 
import os
import pickle
import yaml


params = yaml.safe_load(open('params.ymal'))['model_building']

#fetch 
train_data =  pd.read_csv(os.path.join('data' , 'features' , 'train_bow.csv'))
x_train = train_data.iloc[: , 0:-1]
y_train = train_data.iloc[: , -1]

#define model and train
clf = GradientBoostingClassifier(n_estimators= params['n_estimator'],learning_rate= params['learning_rate'])
clf.fit(x_train , y_train)

#save 
pickle.dump(clf,open('model.pkl' , 'wb'))
