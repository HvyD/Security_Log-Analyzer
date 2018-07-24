
# coding: utf-8

# In[ ]:


from __future__ import print_function
import os
import pandas as pd
import numpy as np
import sklearn


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# # Log Preparations

# In[ ]:



data_path = [pass]
filepath = os.sep.join(data_path + [pass])
data = pd.read_csv(filepath, sep=',')
cleaned_data = data.dropna(axis ='columns', how='all')
sample_data = cleaned_data.head(100000)

for col in sample_data.columns.values.tolist():
    if (len (sample_data[col].value_counts()) == 1) :
        sample_data.drop(col, axis=1, inplace=True)

cleaned_sample_data = sample_data

feature_cols =  ['bytes', 'bytes_in', 'bytes_out',  'ctime', 'reqsize', 'respsize', 'riskscore',  'stime', 'url_length']


strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=20000, random_state=42) # This creates a generator


train_idx, test_idx = next(strat_shuff_split.split(cleaned_sample_data[feature_cols], cleaned_sample_data['action']))

X_train = cleaned_sample_data.loc[train_idx, feature_cols]
y_train = cleaned_sample_data.loc[train_idx, 'action']

X_test = cleaned_sample_data.loc[test_idx, feature_cols]
y_test = cleaned_sample_data.loc[test_idx, 'action']



# # Model Training

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print(accuracy_score(y_test, knn_pred))

