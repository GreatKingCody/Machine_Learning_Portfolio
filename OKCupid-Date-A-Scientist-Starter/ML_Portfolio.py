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





# Write in the csv
df = pd.read_csv('profiles.csv')
# print(df.head())


#Explore the columns
# print(df.columns)
# print(df.head(1))


# print(df.sign.value_counts())
sign = df.sign

split1 = sign.str.split(' ', 1)

signs = []
matter = []


for i in split1:
    if type(i) == list and len(i) == 1:
        signs.append(i)
    elif type(i) == list and len(i) == 2:
        matter.append(i)
# print(signs)
print(matter)