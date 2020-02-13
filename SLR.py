import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
#dataset importing
dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].to_numpy()
y=dataset.iloc[:,3].values
print("X",X)
print("y",y)


#importing missing data
from sklearn.impute import SimpleImputer as Imputer
imputer =Imputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
print("X",X)

#splitting into tain and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].to_numpy()
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#making machine to learn
from sklearn.linear_model import LinearRegression as LR
regressor=LR()
regressor.fit(X_train,y_train)


#predicting the test set results
y_predict=regressor.predict(X_test)

# visualise the training set results
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience Train Set')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience Train Set')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()