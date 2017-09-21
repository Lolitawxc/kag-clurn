import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import json

#take filename as console arg
filename = '2017-09-19-sub1'

with open('experiments/' + filename + '.json') as json_data:
    exp = json.load(json_data)
exp['features'] = sorted(exp['features'])
	
if exp['nrows'] == -1:
	train_csv = pd.read_csv('train.csv')
	train_csv2 = pd.read_csv('transactions.csv')
else:
	train_csv = pd.read_csv('train.csv', nrows = exp['nrows'])	
	train_csv2 = pd.read_csv('transactions.csv', nrows = exp['nrows'])	
train_csv2 = train_csv2.drop_duplicates(subset=['msno'], take_last=True) # un utilisateur apparait plusieurs fois ds les transactions, ce qui crée un prob
test_csv = pd.read_csv('test.csv')	

train_merge = pd.merge(train_csv, train_csv2, on='msno')	
train_merge, val_merge, y_train, y_val = train_test_split(train_merge[exp['features']], train_merge['is_churn'], test_size = 0.2, random_state=42)
test_merge = pd.merge(test_csv, train_csv2, on='msno')	

lr = GradientBoostingClassifier(**exp['params'])
lr.fit(train_merge[exp['features']], y_train)
train_pred = lr.predict_proba(train_merge[exp['features']])[:, 1]
exp['training_score'] = log_loss(y_train, train_pred)
print ('Train logloss')
print (exp['training_score'])
val_pred = lr.predict_proba(val_merge[exp['features']])[:, 1]
exp['val_score'] = log_loss(y_val, val_pred)
print ('Val logloss')
print (exp['val_score'])
test_merge['is_churn'] = lr.predict_proba(test_merge[exp['features']])[:, 1]


test_csv = test_csv.drop(['is_churn'], axis = 1)
res = pd.merge(test_csv, test_merge, on='msno', how='left')
res[['msno', 'is_churn']].fillna(0.5).to_csv('sub', index = False)

with open('experiments/' + filename + '.json', 'w') as json_data:
	json.dump(exp, json_data, indent = 4, sort_keys = True)
