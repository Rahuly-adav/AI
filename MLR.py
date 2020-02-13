import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-2].values
y=dataset.iloc[:,4].values
a=dataset.iloc[:,3]
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
a=labelencoder_X.fit_transform(a)
ohe=OneHotEncoder(categories='auto',dtype=np.float64)
a=ohe.fit_transform(a).toarray()

print("X",X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
s