import joblib

# load the model
model = joblib.load("diabetic_79.pkl")

# Predicting 

result = model.predict([[1,1,1,1,1,1,1,1]])

if result[0]==0:
    print("Person is not Diabetic")
else:
    print("Person is diabetic")
