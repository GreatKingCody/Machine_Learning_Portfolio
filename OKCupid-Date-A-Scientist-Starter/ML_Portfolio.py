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
df.fillna('did not answer', inplace = True)
df = df.convert_dtypes()


#Explore the columns
# print(df.columns)
# print(df.head(1))


# class DataFrame:
    # def __init__(self, df):
    #     self.df = df
     
    # def care(self):
    #     df[['sign', 'care']] = df['sign'].str.split(' ', n = 1, expand = True)
    #     df['sign'] = df['sign'].str.strip('but ')
    #     df['sign'] = df['sign'].str.replace('&rsquo;', '\'')
    #     df['sign'] = df['sign'].astype('str')
    #     df['sign'] = df['sign'].str.replace('None', 'did not answer')
    #     df['sign'] = df['sign'].str.replace('and it matters a lo', 'it matters a lot')
    #     df['sign'] = df['sign'].str.replace('and it\'s fun to think abo', 'it\'s fun to think about')
    #     df['sign'] = df['sign'].str.replace('nan', 'did not answer')

# class_test = DataFrame(df)




def sign(df, column1, column2):
    df[[column1, column2]] = df[column1].str.split(' ', n = 1, expand = True)
    df[column2] = df[column2].str.strip('but ')
    df[column2] = df[column2].str.replace('&rsquo;', '\'')
    df[column2] = df[column2].str.replace('None', 'did not answer')
    df[column2] = df[column2].str.replace('and it matters a lo', 'it matters a lot')
    df[column2] = df[column2].str.replace('and it\'s fun to think abo', 'it\'s fun to think about')
    df[column2] = df[column2].str.replace('nan', 'did not answer')
sign(df, 'sign', 'care')
print(df.care)

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
    df.education = np.where(df.education.str.contains('college', case = False) & ~df.education.str.contains('dropped', case = False), 'graduated from/enrolled in college', df.education)
            

# print(df.education.value_counts())






