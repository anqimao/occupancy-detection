# IoT Project_ occupancy detection
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import *
import operator
import math
from sklearn import tree
import graphviz 

data1 = pd.read_csv("./1.csv", header=0, sep=",", index_col=0)
data2 = pd.read_csv("./2.csv", header=0, sep=",", index_col=0)

# add features
weekday = []
time = []
for i in data1["date"]:
     lin=i.strip().split(" ")
     date_lin=lin[0].split("/")
     str_lin=str(date_lin[0])+str(date_lin[1].zfill(2))+str(date_lin[2].zfill(2))
     sub=pd.to_datetime(str_lin)-pd.to_datetime("20150104")
     weekday.append(int(sub.days%7))
     time.append(int(lin[1].split(":")[0]))
data1["time"] = time
data1["weekday"] = weekday
data1.to_csv('./1.csv')

# detect missing data
print(data1.isna().any())
print(data1.isnull().any())

# outlier detection
name = ['Humidity', 'Light', 'CO2', 'HumidityRatio', 'Temperature']
for j in name:
     a = data1[j]
#     #box plot
#     #plt.boxplot(a)
#     #plt.show()
statistics = a.describe()
statistics.loc['IQR'] = statistics.loc['75%']-statistics.loc['25%'] 
num_lin = 0
for j1 in a:
        if j1>statistics["IQR"]*1.5+statistics["75%"] or j1<-statistics["IQR"]*1.5+statistics["25%"]:
             data1[j][num_lin]=statistics["mean"]
        else:
            pass

# normalization
name = ['Humidity', 'Light', 'CO2', 'HumidityRatio', 'Temperature']
for i in name:
    df_nor = data1[i]
    data1[i] = (df_nor - df_nor.min()) / (df_nor.max() - df_nor.min())
    df_nor1 = data2[i]
    data2[i] = (df_nor1 - df_nor1.min()) / (df_nor1.max() - df_nor1.min())

X_train = np.array(data1[['Humidity', 'Light', 'CO2', 'HumidityRatio', 'weekday', 'time', 'Temperature']])
Y_train = np.array(data1['Occupancy'])
X_test = np.array(data2[['Humidity', 'Light', 'CO2', 'HumidityRatio', 'weekday', 'time', 'Temperature']])
Y_test = np.array(data2['Occupancy'])

# svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)
clf_y_predict = clf.predict(X_test)
print('Accuracy of SVM Classifier:', clf.score(X_test, Y_test))
print("Classification report for %s" % clf)
print(metrics.classification_report(Y_test, clf_y_predict))
print("Confusion matrix")
print(metrics.confusion_matrix(Y_test, clf_y_predict))

# Compute ROC curve and ROC area for each class
y_score = clf.fit(X_train, Y_train).decision_function(X_test)
fpr, tpr, threshold = roc_curve(Y_test, y_score)
roc_auc = auc(fpr, tpr)

lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='orange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for linear SVM')
plt.legend(loc="lower right")
plt.show()

# knn
knn = KNeighborsClassifier(n_neighbors=125, p=5, metric='minkowski')
knn.fit(X_train, Y_train)
knn_y_predict = knn.predict(X_test)
print('Accuracy of KNN Classifier:', knn.score(X_test, Y_test))
neighbors = np.arange(10, 200)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    train_accuracy[i] = knn.score(X_train, Y_train)
    test_accuracy[i] = knn.score(X_test, Y_test)

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# decision tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
dtc_y_predict = dtc.predict(X_test)
print('Accuracy of DTC Classifier:', dtc.score(X_test, Y_test))


y_scored = dtc.fit(X_train, Y_train).predict(X_test)
fpr, tpr, threshold = roc_curve(Y_test, y_scored)
roc_auc = auc(fpr, tpr)

lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for DT')
plt.legend(loc="lower right")
plt.show()
