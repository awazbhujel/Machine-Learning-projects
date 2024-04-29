import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Downloading NLTK data
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('punkt')       # Downloading tokenizer data

vectorizer = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

def transform_text(text):
  text = text.lower()
  text =nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

#function to make prediction
def predict_spam(message):
    transformed_text = transform_text(message)
    message_vectorized = vectorizer.transform([transformed_text])
    prediction = model.predict(message_vectorized)
    return prediction[0]


st.title("Email/SMS Spam Classifier")

st.write("Enter a message to determine if its a spam or not spam:")

#input text box for user to enter message
user_input=st.text_input("Enter message here:")

#button to tigger prediction
if st.button("predict"):
    if user_input:
        prediction = predict_spam(user_input)
        if prediction == 1:
            st.write("This message is Spam")
        else:
            st.write("This Message is not Spam")
    else:
        st.write("please enter a Message")
