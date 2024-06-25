import streamlit as st
from main import cv,clean
import joblib
model = joblib.load('/Users/ronitkhurana/PycharmProjects/CL_NLP_Project/project/model.pkl')

st.title("Tweet Classifier")

user_tweet_input = st.text_input("Enter a tweet to classify:")

if user_tweet_input:
  user_tweet = user_tweet_input
  cleaned_tweet = clean(user_tweet)
  new_data = cv.transform([cleaned_tweet])
  prediction = model.predict(new_data)[0]
  st.write(f"Predicted Sentiment: {prediction}")
  if prediction == "Hate Speech":
      st.warning("This tweet is classified as Hate Speech.")
  elif prediction == "Offensive language":
      st.warning("This tweet contains Offensive Language.")
  else:
      st.success("This tweet does not contain hate speech or offensive language.")
