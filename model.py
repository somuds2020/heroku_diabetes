# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\diabetes\diabetes.csv')

data[['Glucose','BMI']] = data[['Glucose','BMI']].replace(0, np.NaN)
data.dropna(inplace=True)

X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
          'DiabetesPedigreeFunction','Age']].values
y = data[['Outcome']].values

sc = StandardScaler()
X = sc.fit_transform(X)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

logreg = LogisticRegression()
# fit the model with data
logreg.fit(X, y)

# Saving model to disk
pickle.dump(logreg, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\diabetes\model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\diabetes\model.pkl','rb'))
#print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.625, 50]]))

test_x = np.array([[6, 148, 72, 35, 0, 33.6, 0.625, 50]])
test_sc = StandardScaler()
test_x = sc.fit_transform(test_x)
print(model.predict(test_x))