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




# Write in the csv
df = pd.read_csv('profiles.csv')
# print(df.info)


#Explore the columns
# print(df.columns)
# print(df.head(1))


df[['sign', 'care']] = df['sign'].str.split(' ', n = 1, expand = True)

df.care = df.care.str.strip('but ')

df.care = df.care.str.replace('&rsquo;', '\'')

df.care = df.care.astype('str')

df.care = df.care.str.replace('None', 'did not answer')

df.care = df.care.str.replace('and it matters a lo', 'it matters a lot')

df.care = df.care.str.replace('and it\'s fun to think abo', 'it\'s fun to think about')

df.care = df.care.str.replace('nan', 'did not answer')

df = df.convert_dtypes()
# print(df.care.head())

# df.astype({'care': 'str'})

# print(df.columns)


df.body_type.replace(['a little extra', 'curvy', 'full figured', 'overweight'], 'above average', inplace = True)
df.body_type.replace(['fit', 'jacked'], 'athletic', inplace = True)
df.body_type.replace(['skinny', 'used up', 'thin'], 'below average', inplace = True)
df.body_type = df.body_type.str.replace('<NA>', 'did not answer')
df.body_type.replace('rather not answer', 'did not answer', inplace = True)
df.body_type.fillna('did not answer', inplace = True)


df.drinks.replace('desperately', 'very often', inplace = True)
df.drinks.fillna('did not answer', inplace = True)


df.drugs.fillna('did not answer', inplace = True)


print(df.drugs.unique())







