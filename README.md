# IBM-HR-Analytics-Employee-Attrition-Performance
# Dataset
Data set presents an employee survey from IBM, indicating if there is attrition or not. The data set contains approximately 1500 entries. Given the limited size of the data set, the model should only be expected to provide modest improvement in indentification of attrition vs a random allocation of probability of attrition.

# Read Dataset
data=pd.read_csv('Dataset/HR_Analytics.csv')

#Label encoding data
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
y=le.fit_transform(data['Attrition'])
columns=['Gender','Over18','OverTime']
for i in columns:
    df[i]=le.fit_transform(df[i])

colum=['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']
for i in colum:
    df[i]=le.fit_transform(df[i])
    df[i]=le.inverse_transform(df[i])

# importing one hot encoder from sklearn 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
 
# creating one hot encoder object by default 
# entire data passed is one hot encoded 
onehotencoder = OneHotEncoder()

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,3,6,14,16])],remainder='passthrough') # Apply tranformation to change education to binary vector
x = np.array(ct.fit_transform(x))
print(x[:10,:])

# spliting and normalize data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Normalize the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Logistic Regression
Logistic Regression: Logistic Regression is a method similar to linear regression except that the dependent variable is discrete (e.g., 0 or 1). Linear logistic regression estimates the coefficients of a linear model using the selected independent variables while optimizing a classification criterion. For example, this is the logistic regression parameters for our data
from sklearn import linear_model,metrics 
# create logistic regression object 
reg = linear_model.LogisticRegression() 
   
# train the model using the training sets 
reg.fit(x_train, y_train) 
  
# making predictions on the testing set 
y_pred = reg.predict(x_test) 
   
# comparing actual response values (y_test) with predicted response values (y_pred) 
print("Logistic Regression model accuracy(in %):",  metrics.accuracy_score(y_test, y_pred)*100) 
# confusion matrix and Classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# linear regression and random forest

# create linear regression object 
rege = linear_model.LinearRegression() 
   
# train the model using the training sets 
rege.fit(x_train, y_train) 
  
# making predictions on the testing set 
y_pred = rege.predict(x_test)


print("Linear Regression model accuracy(in %):",  metrics.accuracy_score(y_test, y_pred.round())*100)
print(confusion_matrix(y_test, y_pred.round()))
print(classification_report(y_test, y_pred.round()))


# Random Forest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=25, random_state=0)
# train the model using the training sets 
clf.fit(x_train, y_train)

# making predictions on the testing set 
y_pred = clf.predict(x_test)


print("Random Forest model accuracy(in %):",  metrics.accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))





