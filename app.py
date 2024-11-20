import streamlit as st
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()

def transform_email(text):
    # Lower case
    text = text.lower()

    # Tokenization
    text = nltk.wordpunct_tokenize(text)

    # removing special characters
    list = []
    for i in text:
        if i.isalnum():
            list.append(i)

    # Removing stop words and punctuation
    text = list[:]
    list.clear()

    for i in text:
        if i not in (stopwords.words("english") or string.punctuation):
            list.append(i)

    # Stemming
    text = list[:]
    list.clear()

    for i in text:
        list.append(ps.stem(i))

    return " ".join(list)

tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("Email Spam Classsifier")

input_email = st.text_input("Enter the email message")

if st.button("Predict"):

    # Preprocess
    transformed_email = transform_email(input_email)

    # Vectorize
    vector = tfidf.transform([transformed_email])

    # Predict
    result = model.predict(vector)[0]

    # Display
    if result == 2:
        st.header("Spam Email")
        
    else:
        st.header("Ham Email")
    
    


