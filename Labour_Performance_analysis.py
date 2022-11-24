### For data manipulations
import pandas as pd
import numpy as np

### For data Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

### Let's Read the Dataset for Analysis The Productivity
data = pd.read_csv(r"E:\DESKTOPFILES\suraj\labour productivity dataset\labourIOT sensors\IoT Sensors.csv")

### To Know the Information of the Dataset
data.info()

#### let's check the shape of the Dataset
print("Shape of the Dataset:", data.shape)

### Let's check the Head of the Dataset
data.head()

### To know the Names of Each Column in Dataset
data.columns

data.hist(figsize=(14,14), xrot=45)
plt.show()

data.describe()

data.describe(include='object')

for column in data.select_dtypes(include='object'):
    if data[column].nunique() < 10:
        sns.countplot(y=column, data=data)
        plt.show()

data1=data.select_dtypes(exclude=['object'])

for column in data1:
        plt.figure(figsize=(10,8))
        sns.boxplot(data=data1, x=column)
        plt.show()


sns.boxplot(data=data, x='Age', y='Performance/KPI')

plt.figure(figsize=(15,12))
sns.boxplot(data=data, x='Age', y='Body Temperature')
plt.show()

sns.boxplot(data=data, x='Gender', y='Beats Per Minute')

plt.figure(figsize=(25,21))
sns.boxplot(data=data, x='Age', y='Qualification')
plt.show()

### Let's Check the Age present in this Dataset
data['Age'].value_counts()
### With the Help of Value-counts we can see that at the age of 44 there are 20 Workers are working ,at the age of 30 there are 17 Workers are Working, and  
### at the age of 25 there are 16 workers are working......... 


### Let's Check the Gender present in this Dataset
data['Gender'].value_counts()
### With the Help of Value-counts we can see that the Male Workers are 157 working at Site,
#### and The Female Workers are 143 Working at Site.............


### Let's Check the Nationality present in this Dataset
data['Nationality'].value_counts()
### With the Help of Value-counts we can see that the Maximum Workers are from Chine=205 and working at Site,
#### and The Second Highest Workers are 54 from Philippines and Working at Site.............


### Let's Check the Designation present in this Dataset
data['Designation'].value_counts()
### With the Help of Value-counts we can see that the Male Workers are 157 working at Site,
#### and The Female Workers are 143 Working at Site.............


### Let's Check the Qualification present in this Dataset
data['Qualification'].value_counts()
### With the Help of Value-counts we can see that the Male Workers are 157 working at Site,
#### and The Female Workers are 143 Working at Site.............


#### To know the stats about the dataset
data.describe()


data.columns


sns.catplot(data=data, x="Gender", y="Age", kind="box")


sns.catplot(data=data, x="Gender", y="Experience", kind="box")


sns.catplot(data=data, x="Gender", y="Heart Beat", kind="box")


sns.catplot(data=data, x="Gender", y="Beats Per Minute", kind="box")


sns.catplot(data=data, x="Gender", y="Body Temperature", kind="box")


data.columns

data.drop(['Total Working Hours'],axis=1, inplace=True)

sns.pairplot(data)


plt.figure(figsize=(10,7))
data['Gender'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(15,11))
data['Age'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(10,9))
data['Nationality'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(10,10))
data['Designation'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(10,9))
data['Experience'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(15,11))
data['Performance/KPI'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()

plt.figure(figsize=(15,11))
data['Site'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(10,7))
data['Attendance'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(10,7))
data['Motion Indication'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(9,6))
data['Gas Sensor'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(9,6))
data['Noise Detection'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


plt.figure(figsize=(9,6))
data['Infrared Sensor'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


sns.barplot(data['Gender'], data['Age'])
plt.show()


sns.barplot(data['Age'], data['Nationality'])
plt.show()


sns.barplot(data['Gender'], data['Heart Beat'])
plt.show()


sns.barplot(data['Gender'], data['Body Temperature'])
plt.show()


sns.barplot(data['Age'], data['Motion Indication'])
plt.show()


sns.barplot(data['Experience'], data['Gender'], hue = data["Gender"])
plt.show()


plt.figure(figsize=(9,6))
sns.boxplot(data['Age'], data["Nationality"], data["Gender"])
plt.show()


plt.figure(figsize=(16,10))
sns.boxplot(data['Age'], data["Nationality"], data["Designation"])
plt.show()


plt.figure(figsize=(17,10))
sns.boxplot(data['Age'], data["Nationality"], data["Attendance"])
plt.show()

data.columns

sns.set_theme()
f, ax = plt.subplots(figsize=(12,10))
corr = data.corr()
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values,cmap='spring_r')


import sweetviz as sv
report=sv.analyze(data)
report.show_html('sweet_report.html')

from pandas_profiling import ProfileReport
profile=ProfileReport(data,explorative=True)
profile.to_file('output.html')

plt.figure(figsize=(15, 10))
sns.countplot(x=data['Performance/KPI'],data=data)
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x=data['Nationality'],data=data)
plt.show()

plt.figure(figsize=(15,9))
sns.countplot(x=data['Age'],data=data)
plt.show()

plt.figure(figsize=(15,9))
sns.countplot(x=data['Gender'],data=data)
plt.show()

plt.figure(figsize=(15,9))
sns.countplot(x=data['Site'],data=data)
plt.show()

plt.figure()
fig, ax = plt.subplots(figsize=(10, 7))
sns.stripplot(x = "Nationality",
              y = "Age",
              data = data,
              jitter = True,
              ax = ax,
              s = 8)
sns.despine(right = True)
plt.show()


data.isna().sum()

duplicate = data.duplicated()
duplicate
sum(duplicate)

data.columns

data.drop(['Employee ID','Name','RFID Tag','Time In','Time Out','Place','Qualification'],axis=1 ,inplace=True)

data.dtypes

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

data.info()

for i in (1,2,3,7,8,9,11,14,15,16,17,18):
    data.iloc[:,i] = lb.fit_transform(data.iloc[:,i])
data.head()

data.var()

data.skew()

data.kurt()


data.info()

data.shape

data.corr()

data.info()


X = data.loc[:, data.columns!="Performance/KPI"]
X

Y = data["Performance/KPI"]
Y

# splitting the data into testing and training data.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 100)

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions'])


from sklearn.model_selection import KFold





#logistic  regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

lr = LogisticRegression()

lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

lr_train_acc = accuracy_score(Y_train, lr.predict(X_train))
lr_test_acc = accuracy_score(Y_test, Y_pred)

print(f"Training Accuracy of Logistic Regression Model is {lr_train_acc}")
print(f"Test Accuracy of Logistic Regression Model is {lr_test_acc}")


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
#decision tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

Y_pred = dtc.predict(X_test)

dtc_train_acc = accuracy_score(Y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(Y_test, Y_pred)

print(f"Training Accuracy of Decision Tree Model is {dtc_train_acc}")
print(f"Test Accuracy of Decision Tree Model is {dtc_test_acc}")


#random forest
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion = 'gini', max_depth = 8, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 5, n_estimators = 150)
rand_clf.fit(X_train, Y_train)

Y_pred = rand_clf.predict(X_test)

rand_clf_train_acc = accuracy_score(Y_train, rand_clf.predict(X_train))
rand_clf_test_acc = accuracy_score(Y_test, Y_pred)

print(f"Training Accuracy of Random Forest Model is {rand_clf_train_acc}")
print(f"Test Accuracy of Random Forest Model is {rand_clf_test_acc}")

from xgboost import XGBClassifier

xgb = XGBClassifier(booster = 'gblinear', learning_rate = .1, max_depth = 3, n_estimators = 750)
xgb.fit(X_train, Y_train)

Y_pred = xgb.predict(X_test)

xgb_train_acc = accuracy_score(Y_train, xgb.predict(X_train))
xgb_test_acc = accuracy_score(Y_test, Y_pred)

print(f"Training Accuracy of XGB Model is {xgb_train_acc}")
print(f"Test Accuracy of XGB Model is {xgb_test_acc}")

models = [ 'KNN', 'logistic  regression', 'Decision Tree', 'Random Forest', 'XgBoost'] 
scores = [ accuracy_score(Y_test, pred), lr_test_acc, dtc_test_acc, rand_clf_test_acc, xgb_test_acc]

models_test = pd.DataFrame({'Model' : models, 'Score' : scores})


models_test.sort_values(by = 'Score', ascending = False)

plt.figure(figsize = (10, 8))

sns.barplot(x = 'Model', y = 'Score', data = models_test)
plt.show()

models = [ 'KNN', 'logistic  regression', 'Decision Tree', 'Random Forest', 'XgBoost'] 
scores = [  accuracy_score(Y_train, pred_train), lr_train_acc, dtc_train_acc, rand_clf_train_acc, xgb_train_acc]

models_train = pd.DataFrame({'Model' : models, 'Score' : scores})


models_train.sort_values(by = 'Score', ascending = False)

plt.figure(figsize = (10, 8))

sns.barplot(x = 'Model', y = 'Score', data = models_train)
plt.show()