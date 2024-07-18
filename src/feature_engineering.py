import numpy as np
import os 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer

#featch the data from data/preocessed
train_data = pd.read_csv(os.path.join('data' , 'processed' , 'train_processed.csv'))
test_data = pd.read_csv(os.path.join('data' ,'processed' , 'test_processed.csv'))

#remove missing values
train_data.fillna('', inplace = True)
test_data.fillna('',inplace = True)


if train_data.shape[1] > 2 :
    train_data = train_data[['sentiment' , 'content']]
    test_data = train_data[['sentiment' , 'content']]


X_train = train_data['content'].values
y_train = train_data['sentiment'].values
X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features= 500)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)


train_df = pd.DataFrame(X_train_bow.toarray()) 
train_df['label'] = y_train
test_df = pd.DataFrame(X_train_bow.toarray())
test_df['label']=y_test

#store 
path = os.path.join('data' , 'features')
os.makedirs(path,exist_ok= True)



train_df.to_csv(os.path.join(path  , 'train_bow.csv'))
test_df.to_csv(os.path.join(path , 'test_bow.csv'))

