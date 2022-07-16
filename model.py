# This file is showing how model process data form raw to training,
# for modelling there is saperate file
# importing Required libraries
import logging
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from collections import Counter
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# log file initialization 
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.debug(' Model.py File execution started ')

# loading database with pandas library
df = pd.read_csv("./dataset/train.csv")
logging.debug(' Database Loaded ')

df = df.drop(['sl_no','salary'], axis=1)
df = df.apply(lambda x: x.fillna(0))
col_names = df.columns
category_col = ['ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status']

labelEncoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in category_col:
	df[col] = labelEncoder.fit_transform(df[col])

	le_name_mapping = dict(zip(labelEncoder.classes_,
							labelEncoder.transform(labelEncoder.classes_)))

	mapping_dict[col] = le_name_mapping

logging.debug('Database Pre-processing is Finished')

# model featuring
X = df[['gender',
 'ssc_p',
 'ssc_b',
 'hsc_p',
 'hsc_b',
 'hsc_s',
 'degree_p',
 'degree_t',
 'workex',
 'etest_p',
 'specialisation',
 'mba_p']]
y = df['status']

# Data Spliting For model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE = SMOTE()

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))

# model fitting using LGBMClassifier
clf = lgb.LGBMClassifier()
clf.fit(X_train_SMOTE, y_train_SMOTE)

# Printing Accuracy
predictions_e = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions_e))


# pkl export & finish log
pickle.dump(clf, open("model.pkl", "wb"))
logging.debug(' Execution of Model.py is finished ')