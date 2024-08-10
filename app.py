import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')




def text_preprocess(text):

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


  ps = PorterStemmer()
  for i in text:
    y.append(ps.stem(i))
  text = y[:]
  y.clear()


  return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Test prediction
sample_text = "Congratulations! You've won a free ticket."
transformed_text = tfidf.transform([sample_text])
print(model.predict(transformed_text))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter your message")
btn = st.button("Predict")
if btn and len(input_sms)> 0:
  # 1. Preprocess
  transformed_sms = text_preprocess(input_sms)
  # 2. Vectorize
  vectorized_input = tfidf.transform([transformed_sms])
  # 3. Predict
  result = model.predict(vectorized_input)[0]
  # 4. Display
  if result == 1:
    st.header(":red[Spam]")
  else:
    st.header(":green[Not Spam]")



