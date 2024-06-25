import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
import string
import joblib
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words("english"))



def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?]', '', text)
    text = re.sub('https?://\S+|www\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


df = pd.read_csv("/Users/ronitkhurana/PycharmProjects/CL_NLP_Project/project/data/twitter_data.csv")
print(df.shape)
class_0 = df[df['class'] == 0]
class_1 = df[df['class'] == 1]
class_2 = df[df['class'] == 2]

class_0 = class_0.sample(n=1430, replace=True, random_state=20)
class_1 = class_1.sample(n=1800, replace=True, random_state=20)
class_2 = class_2.sample(n=1430, replace=True, random_state=20)

df = pd.concat([class_0, class_1, class_2], ignore_index=True)

df['labels'] = df['class'].map({0: "Hate Speech", 1: "Offensive language", 2: "No hate speech"})

cv = CountVectorizer()


def train_save_model(X_train, X_test, y_train, y_test):
    cv.fit(X_train)

    X_train = cv.transform(X_train)
    X_test = cv.transform(X_test)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print("Accuracy: ", clf.score(X_test, y_test))

    joblib.dump(clf, 'model.pkl')


x = np.array(df["tweet"])
y = np.array(df["labels"])



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=25)
train_save_model(X_train, X_test, y_train, y_test)
