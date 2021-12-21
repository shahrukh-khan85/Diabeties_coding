import pickle

model = pickle.load(open("diabetic_79.sav","rb"))


result = model.predict([[1,1,1,1,1,1,1,1]])

if result[0]==0:
    print("Person is not diabetic")
else:
    print("Person is diabetic")