import joblib
from main import cv


clf = joblib.load('model.pkl')


print("\n\n***TESTING HATE SPEECH DETECTION MODEL***\n")

while True:

    test_data=input("Enter tweet to detect HateSpeech/offensive language: ")
    df=cv.transform([test_data]).toarray()
    print(clf.predict(df))

