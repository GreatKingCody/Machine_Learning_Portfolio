import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from yaml import ScalarEvent




# Write in the csv
df = pd.read_csv('profiles.csv')
df.fillna('did not answer', inplace = True)
df = df.convert_dtypes()





def sign(df, column1, column2):
    df[[column1, column2]] = df[column1].str.split(' ', n = 1, expand = True)
    df[column1] = df[column1].astype('str')
    df[column2] = df[column2].astype('str')
    df[column2] = df[column2].str.strip('but ')
    df[column2] = df[column2].str.replace('&rsquo;', '\'')
    df[column2] = df[column2].str.replace('and it matters a lo', 'it matters a lot')
    df[column2] = df[column2].str.replace('and it\'s fun to think abo', 'it\'s fun to think about')
    df[column2] = df[column2].str.replace('<NA>', 'did not answer')
    df[column2] = df[column2].str.replace(r'^not answer$', 'did not answer', regex = True)
sign(df, 'sign', 'care')

def body_type(df, column):
    df[column].replace(['a little extra', 'curvy', 'full figured', 'overweight'], 'above average', inplace = True)
    df[column].replace(['fit', 'jacked'], 'athletic', inplace = True)
    df[column].replace(['skinny', 'used up', 'thin'], 'below average', inplace = True)
    df[column].replace('rather not answer', 'did not answer', inplace = True)
df.body_type = body_type(df, 'body_type')

def drinks(df, column):
    df[column].replace('desperately', 'very often', inplace = True)
df.drinks = drinks(df, 'drinks')

def education(df, column):
    df[column] = np.where(df[column].str.contains('college', case = False) & ~df[column].str.contains('dropped', case = False), 'graduated from/enrolled in college', df[column])
    df[column] = np.where(df[column].str.contains('masters', case = False) & ~df[column].str.contains('dropped', case = False), 'graduated from/enrolled in a master\'s program', df[column])
    df[column] = np.where(df[column].str.contains('ph.d', case = False) & ~df[column].str.contains('dropped', case = False), 'graduated from/enrolled in a ph.d program', df[column])
    df[column] = np.where(df[column].str.contains('law', case = False) & ~df[column].str.contains('dropped', case = False), 'graduated from/enrolled in law school', df[column])
    df[column] = np.where(df[column].str.contains('high', case = False) & ~df[column].str.contains('dropped', case = False), 'graduated from/enrolled in high school', df[column])
    df[column] = np.where(df[column].str.contains('med', case = False) & ~df[column].str.contains('dropped', case = False), 'graduated from/enrolled in med school', df[column])
    df[column] = np.where(df[column].str.contains('space', case = False) & ~df[column].str.contains('dropped', case = False), 'graduated from/enrolled in space camp', df[column])
    df[column] = np.where(df[column].str.contains('high', case = False) & df[column].str.contains('dropped', case = False), 'dropped out of high school', df[column])
    df[column] = np.where(df[column].str.contains('college', case = False) & df[column].str.contains('dropped', case = False), 'dropped out of higher education', df[column])
    df[column] = np.where(df[column].str.contains('med', case = False) & df[column].str.contains('dropped', case = False), 'dropped out of higher education', df[column])
    df[column] = np.where(df[column].str.contains('law', case = False) & df[column].str.contains('dropped', case = False), 'dropped out of higher education', df[column])
    df[column] = np.where(df[column].str.contains('ph.d', case = False) & df[column].str.contains('dropped', case = False), 'dropped out of higher education', df[column])
    df[column] = np.where(df[column].str.contains('masters', case = False) & df[column].str.contains('dropped', case = False), 'dropped out of higher education', df[column])
    df[column] = np.where(df[column].str.contains('space', case = False) & df[column].str.contains('dropped', case = False), 'dropped out of space camp', df[column])
education(df, 'education')

def income(df, column):
    df[column] = df[column].astype('str')
    df[column] = df[column].str.strip()
    df[column].replace('-1', '0', inplace = True)
    df[column] = df[column].astype('int64')
income(df, 'income')

df.drugs = df.drugs.astype('str')






pre_features = df[['body_type', 'education', 'drinks', 'drugs', 'sign', 'care']]
dummies = pd.get_dummies(pre_features, drop_first = True)
features_table = pd.concat([df.age, dummies], axis = 'columns')

features = features_table.to_numpy()

pre_strip = df.income.astype('str')
strip = pre_strip.str.strip(' ')
labels = strip.astype('int64').to_numpy()
# print(labels)



train_data, test_data, train_labels, test_labels = train_test_split(features, labels, random_state = 11, test_size = 0.2)




model = RandomForestClassifier()

model.fit(train_data, train_labels)


print(model.score(test_data, test_labels))








