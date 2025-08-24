import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import time
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('data:image/jpeg;base64,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');
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("Next Word Prediction Application- Akash Kamble")
#Load the LSTM Model
model=load_model('next_word_lstm.h5')
#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)



# Predict the next word
def predict_next_word(text,n=2):
    for i in range(n):
    # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
        # predict
        pos = np.argmax(model.predict(padded_token_text))

        for word,index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                
            
    return ' '.join(text.split()[-n:])

input_text=st.text_input("Enter the sequence of Words","What is the course")
n = st.number_input("No of Words need to predict:", min_value=0, max_value=25, value=6, step=1)

if st.button("Predict Next Word"):    
    with st.spinner('Loading...'):
        time.sleep(1)
        try:
        # st.write("Loading model and tokenizer...")
        # max_sequence_len = model.input_shape[1]  # Extract max sequence length from model input shape
        
            next_word = predict_next_word(input_text,n)           
            st.markdown(f"{input_text}<span style='color:green; font-size:18px;'> {next_word}</span>", unsafe_allow_html=True)

            #
        except Exception as e:
            st.error(f"Error: {e}")


footer="""<style>

a:hover,  a:active {
color: red;
background-color: transparent;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p> Develop with <span style='color:red;'>❤ </span> by Sky </p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
