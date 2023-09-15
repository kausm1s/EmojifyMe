import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from keras.utils import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt

app_mode = st.sidebar.selectbox('Mode',['Home'])

option = st.selectbox('Select model type:', ('LSTM', 'BiLSTM','RNN'))

if option=='LSTM': 
    model = pickle.load(open('C:/Users/kaust/OneDrive/Desktop/dlops_profolder/model_lstm.pkl','rb'))

elif option=='BiLSTM':
    model = pickle.load(open('C:/Users/kaust/OneDrive/Desktop/dlops_profolder/model.pkl','rb'))

elif option=='RNN':
    model = pickle.load(open('C:/Users/kaust/OneDrive/Desktop/dlops_profolder/model_rnn.pkl','rb'))

tokenizer = pickle.load(open("C:\\Users\\kaust\\OneDrive\\Desktop\\dlops_profolder\\tweet_tokenizer",'rb'))

emoji_raw = open('C:\\Users\\kaust\\OneDrive\\Desktop\\dlops_profolder\\us_mapping.txt','r',encoding="utf8")

emojis=[]
for sentence in emoji_raw:
        sentence = sentence.rstrip()
        emojis.append(sentence)

print(emojis)        
emoji_dict={}
all_emojis = ""

for e in emojis:
        idx = int(e.split()[0])
        emoji = e.split()[1]
        emoji_dict[idx] = emoji
        all_emojis+=(emoji)



def preprocess_text(X):
        max_len=40
        X_seqs = tokenizer.texts_to_sequences(X)
        X_seqs_pd = pad_sequences(X_seqs, truncating="pre", padding="pre", maxlen=max_len)
        return X_seqs_pd

def tweet_clean(tweet):
        tweet = str(tweet).lower()
        rm_mention = re.sub(r'@[A-Za-z0-9]+', '', tweet)                       # remove @mentions
        rm_rt = re.sub(r'RT[/s]+', '', rm_mention)                             # remove RT
        rm_links = re.sub(r'http\S+', '', rm_rt)                               # remove hyperlinks
        rm_links = re.sub(r'https?:\/\/\S+','', rm_links)
        rm_nums = re.sub('[0-9]+', '', rm_links)                               # remove numbers
        rm_punc = [char for char in rm_nums if char not in string.punctuation] # remove punctuations
        rm_punc = ''.join(rm_punc)
        cleaned = rm_punc
        
        return cleaned


def emoji_prediction(text):

        text = tweet_clean(text)
        X_sequences = preprocess_text([text])
        predictions = np.argmax(model.predict(X_sequences), axis=1)
        # print(predictions)
        emoji_idx = predictions[0]
        emoji = emoji_dict[emoji_idx]
        
        return emoji, model.predict(X_sequences)

def main():

        st.title("Emoji Prediction")
        text = st.text_input('Enter the text for emoji prediction')
        emojis = st.text(' ' + all_emojis)
        prediction = ''
        print(emoji_prediction(text)[1][0])
        print([all_emojis[i] for i in range(len(all_emojis))])

        if st.button('Emoji Prediction Result'):
            prediction = text + " " + emoji_prediction(text)[0]
            
            fig, ax = plt.subplots(figsize=(10, 3))

            # Plot the data as a heatmap
            im = ax.imshow(emoji_prediction(text)[1], cmap='Blues')

            # Add a colorbar
            cbar = ax.figure.colorbar(im, ax=ax)

            # Set the axis labels and title
            ax.set_xticks(np.arange(20))
            ax.set_xticklabels(['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜'])
            ax.set_xlabel('Emojis')
            ax.set_ylabel('')
            ax.set_title('Heatmap of Probability of emojis')

            # Display the plot using Streamlit
            st.pyplot(ax.figure)


        st.success(prediction)

if __name__ == '__main__':
    main()
