import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn

ps=PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tdidf = pickle.load(open('vectorizer1.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title('Fake News Classifier')

input_news=st.text_area("Enter the news article")

if st.button('Predict'):
        # 1. Preprocess

        transformed_news=transform_text(input_news)

        # 2. Vectorize

        vector_input=tdidf.transform([transformed_news])

        # 3. Predict

        result=model.predict(vector_input)[0]
        # 4. Display

        if result==1:
            st.header("Real News")
        else:
            st.header("Fake News")