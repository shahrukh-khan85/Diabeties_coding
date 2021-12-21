import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]

dataframe = pd.read_csv(url, names=names)
print(dataframe)

x = dataframe.iloc[:,:8]
y = dataframe.iloc[:,-1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.20, random_state=101)

# fit the model
model = LogisticRegression()
model.fit(x_train, y_train)

# score
result = model.score(x_test, y_test)
print(result)

# saving the file in the model
joblib.dump(model,"diabetic_79.pkl")
