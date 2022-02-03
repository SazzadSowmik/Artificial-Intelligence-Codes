#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Income Dataset (50k).csv')
#print(dataset.head(5))
#print(dataset.shape)
#print(dataset.isnull().sum())


'''Handling the missing values'''

dataset = dataset.drop(['fnlwgt', 'educational-num', 'relationship', 'marital-status'], axis=1)
dataset = dataset.dropna(axis=0, subset=['occupation'])

from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
impute = impute.fit(dataset[['native-country']])
dataset[['native-country']] = impute.transform(dataset[['native-country']])
#print(dataset.head(74))

'''Encoding Variable'''
#print(dataset['gender'].unique())
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dataset['gender'] = enc.fit_transform(dataset['gender'])
#print(dataset[['gender']].head(10))
occ_enc = pd.get_dummies(dataset['occupation'])
wc_enc = pd.get_dummies(dataset['workclass'])
edu_enc = pd.get_dummies(dataset['education'])

'''if we include the feature the accuracy decreases to 71% from 74%'''
dataset = pd.concat([dataset, occ_enc], axis=1)

dataset = pd.concat([dataset, wc_enc], axis=1)
dataset = pd.concat([dataset, edu_enc], axis=1)
dataset = dataset.drop(['workclass', 'occupation', 'education'], axis=1)
#print(dataset)


'''Spilt Label and feature'''
from sklearn.model_selection import train_test_split
X = dataset
X = X.drop(['age', 'race', 'gender', 'capital-gain', 'capital-loss', 'native-country', 'income_>50K'], axis=1)
y = dataset['income_>50K']
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''scaling'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)

'''accuracy'''
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
accuracy_general = knn.score(X_test, y_test)*100

print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))


# In[29]:


'''LogisticRegression'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

'''accuracy_logisticRegression'''
print("Training accuracy of the model is {:.2f}".format(model.score(X_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(model.score(X_test, y_test)))


# In[30]:


'''DecisionTree'''
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

print("Training accuracy of the model is {:.2f}".format(model.score(X_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(model.score(X_test, y_test)))


# In[31]:


'''Support Vector Classifier before PCA'''
### it takes too long to compile and run ###
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)

from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

print("Training accuracy of the model is {:.2f}".format(svc.score(X_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(X_test, y_test)))


'''
preSVMtrain = 0.75
preSVMtest = 0.76

''' 
preSVMtrain = svc.score(X_train, y_train)
preSVMtest = svc.score(X_test, y_test)


# In[32]:


'''Ensemble Classifier(Random Forest) before PCA'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train, y_train)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(X_test, y_test)))

preRFCtrain = rfc.score(X_train, y_train)
preRFCtest = rfc.score(X_train, y_train)


# In[33]:


'''Neural Network Classifier before PCA'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)


from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)
nnc.fit(X_train, y_train)

print("The Training accuracy of the model is {:.2f}".format(nnc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(X_test, y_test)))

preNNCtrain = nnc.score(X_train, y_train)
preNNCtest = nnc.score(X_train, y_train)


# In[34]:


'''PCA'''
from sklearn.decomposition import PCA 
pca = PCA(n_components=23)


principal_components= pca.fit_transform(X)
X = principal_components


# In[ ]:


'''Support Vector Classifier after PCA'''
### it takes too long to compile and run ###
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)

from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

print("Training accuracy of the model is {:.2f}".format(svc.score(X_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(X_test, y_test)))


'''
postSVMtrain = 0.79
postSVMtest = 0.79

'''
postSVMtrain = svc.score(X_train, y_train)
postSVMtest = svc.score(X_test, y_test)


# In[ ]:


'''Ensemble Classifier(Random Forest) after PCA'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train, y_train)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(X_test, y_test)))

postRFCtrain = rfc.score(X_train, y_train)
postRFCtest = rfc.score(X_train, y_train)


# In[ ]:


'''Neural Network Classifier after PCA'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)


from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)
nnc.fit(X_train, y_train)

print("The Training accuracy of the model is {:.2f}".format(nnc.score(X_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(X_test, y_test)))

postNNCtrain = nnc.score(X_train, y_train)
postNNCtest = nnc.score(X_train, y_train)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
data = [[preSVMtest, preRFCtest, preNNCtest],
[postSVMtest, postRFCtest, postNNCtest],
       [preSVMtrain, preRFCtrain, preNNCtrain],
       [postSVMtrain, postRFCtrain, postRFCtrain]]
X = np.arange(3)
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Accuracy', fontweight ='bold', fontsize = 12)
ax.set_title('Income dataset')

ax.bar(X + 0.00, data[0], color = 'b', width = 0.10)
ax.bar(X + 0.1, data[1], color = 'g', width = 0.10)
ax.bar(X + 0.3, data[2], color = 'r', width = 0.10)
ax.bar(X + 0.4, data[2], color = 'y', width = 0.10)

ax.set_xlabel('Support Vector Machine                                              Random Forest                                    Neural Network')


colors = {'pre PCA test':'b', 'post PCA test':'g', 'pre PCA train':'r', 'post PCA train' : 'y'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.show()


# In[ ]:




