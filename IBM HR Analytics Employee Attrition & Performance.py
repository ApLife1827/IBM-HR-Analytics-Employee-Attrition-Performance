import pandas as pd
import numpy as np


#Read Dataset
data=pd.read_csv('Dataset/HR_Analytics.csv')
print(data.head())

df=data.drop('Attrition', axis=1)

#Label encoding data
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
y=le.fit_transform(data['Attrition'])
columns=['Gender','Over18','OverTime']
for i in columns:
    df[i]=le.fit_transform(df[i])
a=df[:2].values
print(a)

colum=['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']
for i in colum:
    df[i]=le.fit_transform(df[i])
    df[i]=le.inverse_transform(df[i])
x=df[:].values
print(x)

# importing one hot encoder from sklearn 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
 
# creating one hot encoder object by default 
# entire data passed is one hot encoded 
onehotencoder = OneHotEncoder()

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,3,6,14,16])],remainder='passthrough') # Apply tranformation to change education to binary vector
x = np.array(ct.fit_transform(x))
print(x[:10,:])

#Regression Techniques

#logistic regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,jaccard_score,f1_score,log_loss,confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Normalize the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn import linear_model,metrics 
# create logistic regression object 
reg = linear_model.LogisticRegression() 
   
# train the model using the training sets 
reg.fit(x_train, y_train) 
  
# making predictions on the testing set 
y_pred = reg.predict(x_test) 
   
# comparing actual response values (y_test) with predicted response values (y_pred) 
print("Logistic Regression model accuracy(in %):",  metrics.accuracy_score(y_test, y_pred)*100) 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

######################################################################################################

#Linear regression
# create linear regression object 
rege = linear_model.LinearRegression() 
   
# train the model using the training sets 
rege.fit(x_train, y_train) 
  
# making predictions on the testing set 
y_pred = rege.predict(x_test)


print("Linear Regression model accuracy(in %):",  metrics.accuracy_score(y_test, y_pred.round())*100)
print(confusion_matrix(y_test, y_pred.round()))
print(classification_report(y_test, y_pred.round()))

#########################################################################################################

#Random Forest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=25, random_state=0)
# train the model using the training sets 
clf.fit(x_train, y_train)

# making predictions on the testing set 
y_pred = clf.predict(x_test)


print("Random Forest model accuracy(in %):",  metrics.accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

