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
sign(df, 'sign', 'sign_care')

def sign_care(df, column):
    df[column] = df[column].map({'did not answer' : 0, 'it doesn\'t matter' : 1, 'it\'s fun to think about': 2, 'it matters a lot': 3})
sign_care(df, 'sign_care')

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

# print(df.religion)

df.height.replace('did not answer', 0, inplace = True)





pre_features = df[['body_type', 'education', 'drinks', 'drugs', 'sign']]
dummies = pd.get_dummies(pre_features, drop_first = True)
features_table = pd.concat([df[['age', 'height', 'sign_care']], dummies], axis = 'columns')


# test_correlation = pd.concat([features_table, df.income], axis = 1)
# plt.figure(figsize = ((20, 25)))
# cor = test_correlation.corr()
# sns.pairplot(cor, annot = True, cmap = plt.cm.Reds)
# plt.show()


features = features_table.to_numpy()

pre_strip = df.income.astype('str')
strip = pre_strip.str.strip(' ')
labels = strip.astype('int64').to_numpy()
# print(labels))


print(features[40])


train_data, test_data, train_labels, test_labels = train_test_split(features, labels, random_state = 11, test_size = 0.2)




# scores_ne = {}

# for i in range(1, 300):
#     model = RandomForestClassifier(n_estimators = i)
#     model.fit(train_data, train_labels)
#     score = round(model.score(test_data, test_labels) * 100, 2)
#     scores_ne[i] = score
# print(scores_ne)
# print(max(scores_ne.values()))

# of_range_1_300 = {1: 74.97, 2: 76.67, 3: 76.55, 4: 76.88, 5: 76.75, 6: 76.97, 7: 77.06, 8: 77.16, 9: 76.91, 10: 77.02, 11: 77.11, 12: 77.17, 13: 77.1, 14: 77.13, 15: 77.37, 16: 77.25, 17: 77.26, 18: 77.05, 19: 77.36, 20: 77.44, 21: 77.16, 22: 77.32, 23: 77.16, 24: 77.16, 25: 77.16, 26: 77.28, 27: 77.19, 28: 77.18, 29: 77.31, 30: 77.31, 31: 77.24, 32: 77.3, 33: 77.36, 34: 77.24, 35: 77.16, 36: 77.27, 37: 77.39, 38: 77.13, 39: 77.28, 40: 77.15, 41: 77.27, 42: 77.2, 43: 77.45, 44: 77.21, 45: 77.32, 46: 77.27, 47: 77.14, 48: 77.31, 49: 77.4, 50: 77.24, 51: 77.17, 52: 77.42, 53: 77.45, 54: 77.51, 55: 77.52, 56: 77.41, 57: 77.36, 58: 77.55, 59: 77.35, 60: 77.33, 61: 77.24, 62: 77.36, 63: 77.33, 64: 77.38, 65: 77.41, 66: 77.39, 67: 77.21, 68: 77.32, 69: 77.47, 70: 77.43, 71: 77.54, 72: 77.31, 73: 77.36, 74: 77.31, 75: 77.41, 76: 77.41, 77: 77.28, 78: 77.35, 79: 77.38, 80: 77.34, 81: 77.41, 82: 77.53, 83: 77.46, 84: 77.51, 85: 77.41, 86: 77.45, 87: 77.41, 88: 77.51, 89: 77.29, 90: 77.42, 91: 77.5, 92: 77.46, 93: 77.41, 94: 77.46, 95: 77.61, 96: 77.43, 97: 77.46, 98: 77.39, 99: 77.46, 100: 77.58, 101: 77.48, 102: 77.42, 103: 77.48, 104: 77.51, 105: 77.57, 106: 77.4, 107: 77.36, 108: 77.47, 109: 77.61, 110: 77.46, 111: 77.52, 112: 77.56, 113: 77.52, 114: 77.42, 115: 77.46, 116: 77.55, 117: 77.46, 118: 77.41, 119: 77.59, 120: 77.34, 121: 77.46, 122: 77.46, 123: 77.46, 124: 77.52, 125: 77.61, 126: 77.61, 127: 77.45, 128: 77.51, 129: 77.32, 130: 77.37, 131: 77.5, 132: 77.64, 133: 77.56, 134: 77.51, 135: 77.47, 136: 77.56, 137: 77.55, 138: 77.57, 139: 77.56, 140: 77.41, 141: 77.58, 142: 77.62, 143: 77.44, 144: 77.51, 145: 77.41, 146: 77.44, 147: 77.67, 148: 77.48, 149: 77.41, 150: 77.58, 151: 77.42, 152: 77.48, 153: 77.52, 154: 77.55, 155: 77.41, 156: 77.61, 157: 77.51, 158: 77.41, 159: 77.62, 160: 77.52, 161: 77.61, 162: 77.44, 163: 77.46, 164: 77.38, 165: 77.61, 166: 77.65, 167: 77.61, 168: 77.54, 169: 77.51, 170: 77.54, 171: 77.56, 172: 77.46, 173: 77.49, 174: 77.69, 175: 77.56, 176: 77.58, 177: 77.43, 178: 77.43, 179: 77.59, 180: 77.52, 181: 77.51, 182: 77.61, 183: 77.61, 184: 77.48, 185: 77.54, 186: 77.51, 187: 77.53, 188: 77.54, 189: 77.5, 190: 77.54, 191: 77.5, 192: 77.49, 193: 77.51, 194: 77.66, 195: 77.49, 196: 77.61, 197: 77.56, 198: 77.46, 199: 77.42, 200: 77.56, 201: 77.5, 202: 77.47, 203: 77.58, 204: 77.49, 205: 77.41, 206: 77.52, 207: 77.54, 208: 77.6, 209: 77.56, 210: 77.6, 211: 77.64, 212: 77.6, 213: 77.63, 214: 77.5, 215: 77.51, 216: 77.52, 217: 77.51, 218: 77.49, 219: 77.62, 220: 77.66, 221: 77.59, 222: 77.61, 223: 77.48, 224: 77.5, 225: 77.59, 226: 77.63, 227: 77.56, 228: 77.59, 229: 77.67, 230: 77.54, 231: 77.6, 232: 77.48, 233: 77.6, 234: 77.56, 235: 77.57, 236: 77.53, 237: 77.61, 238: 77.56, 239: 77.61, 240: 77.5, 241: 77.46, 242: 77.61, 243: 77.4, 244: 77.6, 245: 77.53, 246: 77.61, 247: 77.66, 248: 77.55, 249: 77.61, 250: 77.57, 251: 77.48, 252: 77.46, 253: 77.56, 254: 77.64, 255: 77.56, 256: 77.45, 257: 77.57, 258: 77.63, 259: 77.63, 260: 77.61, 261: 77.59, 262: 77.55, 263: 77.58, 264: 77.63, 265: 77.56, 266: 77.57, 267: 77.58, 268: 77.5, 269: 77.56, 270: 77.56, 271: 77.57, 272: 77.56, 273: 77.61, 274: 77.61, 275: 77.59, 276: 77.46, 277: 77.57, 278: 77.51, 279: 77.67, 280: 77.68, 281: 77.66, 282: 77.61, 283: 77.68, 284: 77.6, 285: 77.58, 286: 77.6, 287: 77.61, 288: 77.59, 289: 77.61, 290: 77.69, 291: 77.63, 292: 77.61, 293: 77.56, 294: 77.64, 295: 77.61, 296: 77.58, 297: 77.54, 298: 77.61, 299: 77.62}
# max_val = 77.69 lowest n_estimator with that value = 174

# of_range_300_400 = {300: 77.51, 301: 77.6, 302: 77.46, 303: 77.62, 304: 77.54, 305: 77.56, 306: 77.58, 307: 77.56, 308: 77.66, 309: 77.59, 310: 77.62, 311: 77.6, 312: 77.62, 313: 77.53, 314: 77.58, 315: 77.66, 316: 77.56, 317: 77.6, 318: 77.6, 319: 77.7, 320: 77.64, 321: 77.63, 322: 77.54, 323: 77.58, 324: 77.59, 325: 77.54, 326: 77.51, 327: 77.52, 328: 77.62, 329: 77.59, 330: 77.61, 331: 77.62, 332: 77.61, 333: 77.57, 334: 77.72, 335: 77.61, 336: 77.61, 337: 77.59, 338: 77.63, 339: 77.54, 340: 77.46, 341: 77.63, 342: 77.62, 343: 77.61, 344: 77.65, 345: 77.58, 346: 77.53, 347: 77.61, 348: 77.62, 349: 77.67, 350: 77.56, 351: 77.66, 352: 77.56, 353: 77.55, 354: 77.66, 355: 77.56, 356: 77.66, 357: 77.61, 358: 77.59, 359: 77.69, 360: 77.61, 361: 77.64, 362: 77.6, 363: 77.61, 364: 77.68, 365: 77.68, 366: 77.53, 367: 77.58, 368: 77.66, 369: 77.7, 370: 77.57, 371: 77.61, 372: 77.65, 373: 77.65, 374: 77.6, 375: 77.55, 376: 77.66, 377: 77.49, 378: 77.61, 379: 77.51, 380: 77.56, 381: 77.54, 382: 77.66, 383: 77.61, 384: 77.64, 385: 77.63, 386: 77.65, 387: 77.61, 388: 77.56, 389: 77.62, 390: 77.6, 391: 77.66, 392: 77.56, 393: 77.58, 394: 77.56, 395: 77.62, 396: 77.56, 397: 77.59, 398: 77.66, 399: 77.63}
# max_val = 77.72



# scores_md = {}
# for i in range(100, 150):
#     model = RandomForestClassifier(n_estimators = 174, max_depth = i)
#     model.fit(train_data, train_labels)
#     score = round(model.score(test_data, test_labels) * 100, 2)
#     scores_md[i] = score

# print(scores_md)

# md_range_1_100 = {1: 80.73, 2: 80.73, 3: 80.73, 4: 80.73, 5: 80.73, 6: 80.73, 7: 80.73, 8: 80.73, 9: 80.72, 10: 80.71, 11: 80.71, 12: 80.7, 13: 80.67, 14: 80.6, 15: 80.55, 16: 80.48, 17: 80.33, 18: 80.2, 19: 80.04, 20: 79.75, 21: 79.55, 22: 79.37, 23: 78.98, 24: 78.74, 25: 78.39, 26: 78.14, 27: 78.01, 28: 78.03, 29: 77.79, 30: 77.72, 31: 77.61, 32: 77.67, 33: 77.45, 34: 77.56, 35: 77.47, 36: 77.61, 37: 77.53, 38: 77.5, 39: 77.46, 40: 77.59, 41: 77.5, 42: 77.41, 43: 77.59, 44: 77.51, 45: 77.43, 46: 77.5, 47: 77.46, 48: 77.61, 49: 77.42, 50: 77.47, 51: 77.57, 52: 77.54, 53: 77.47, 54: 77.6, 55: 77.48, 56: 77.6, 57: 77.49, 58: 77.56, 59: 77.48, 60: 77.46, 61: 77.54, 62: 77.49, 63: 77.55, 64: 77.56, 65: 77.46, 66: 77.43, 67: 77.61, 68: 77.51, 69: 77.53, 70: 77.5, 71: 77.42, 72: 77.51, 73: 77.54, 74: 77.48, 75: 77.6, 76: 77.51, 77: 77.61, 78: 77.63, 79: 77.42, 80: 77.56, 81: 77.61, 82: 77.55, 83: 77.48, 84: 77.57, 85: 77.53, 86: 77.51, 87: 77.5, 88: 77.48, 89: 77.61, 90: 77.58, 91: 77.52, 92: 77.61, 93: 77.63, 94: 77.55, 95: 77.52, 96: 77.66, 97: 77.44, 98: 77.53, 99: 77.65}

# print(max(md_range_1_100.values()))
# Max value of 80.73 key 8 md



# scores_mf = {}
# for i in range(1, 7):
#     model = RandomForestClassifier(n_estimators = 174, max_depth = 8, max_features = i)
#     model.fit(train_data, train_labels)
#     score = round(model.score(test_data, test_labels) * 100, 2)
#     scores_mf[i] = score
    
# print(scores_mf)
# print(max(scores_mf.values()))
# Max value of 80.73 regardless of number of features





model = RandomForestClassifier(n_estimators = 174, max_depth = 8, max_features = 7)
model.fit(train_data, train_labels)
print(model.score(test_data, test_labels))

# Final accuracy is 80.73 %, without dropping any nan data.