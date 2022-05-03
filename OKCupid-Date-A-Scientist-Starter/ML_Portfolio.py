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
print(df.info)


#Explore the columns
# print(df.columns)
# print(df.head(1))



# Binarise labels
mlb = MultiLabelBinarizer()
expandedLabelData = mlb.fit_transform(df["sign"])
labelClasses = mlb.classes_


# Create a pandas.DataFrame from our output
expandedLabels = pd.DataFrame(expandedLabelData, columns=labelClasses)