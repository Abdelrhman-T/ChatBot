from flask import Flask, render_template, request, jsonify
from groq import Groq

import numpy as np
import pandas as pd
import re
import spacy
from num2words import num2words
import tensorflow as tf
from transformers import AutoTokenizer
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras.models import load_model


key = "61b9e78256564ea68ff634e42eab3289"
endpoint = "https://language-servieses.cognitiveservices.azure.com/"

text_analytics_clint = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

app = Flask(__name__)

from tensorflow.keras.preprocessing.text import Tokenizer

df_process = pd.read_csv("models/df_process.csv")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_process["input_texts processed"]+df_process["target_texts processed"])

nlp = spacy.load("en_core_web_sm")

max_sequence_len_encoder, max_sequence_len_decoder = 14, 61
total_words = len(tokenizer.word_index) + 1

def convert_time_to_words(time_str):
    time_parts = time_str.split(":")
    hours = num2words(time_parts[0])
    minutes = num2words(time_parts[1])
    time_in_words = f"{hours} {minutes}"
    return time_in_words

def preprocess_text(text):
    text = re.sub(r'\d{1,2}:\d{2}', lambda x: convert_time_to_words(x.group()), text)
    text = re.sub(r"\d+", lambda x: num2words(int(x.group())), text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text


seq2seq = load_model("models\chatBot_model.h5")


# Inference Encoder Model
encoder_inputs = seq2seq.input[0]
encoder_outputs, state_h_enc, state_c_enc = seq2seq.layers[4].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model([encoder_inputs], encoder_states)

# Inference Decoder Model
decoder_inputs = Input(shape=(None,))
decoder_state_input_h = Input(shape=(500,))
decoder_state_input_c = Input(shape=(500,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding = seq2seq.layers[3]
decoder_embedding_outputs = decoder_embedding(decoder_inputs)

decoder_lstm = seq2seq.layers[5]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding_outputs, initial_state=decoder_state_inputs
)
decoder_states = [state_h, state_c]

decoder_dense = seq2seq.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)



def chatBot(question):
    # Preprocess the input text
    input_text = preprocess_text(question)  # Assuming this is your custom preprocessing function

    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_len_encoder, padding='post')

    # Convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_sequence)

    # Get encoder states
    stat = encoder_model.predict(input_tensor, verbose=0)

    # Prepare for decoding
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index.get("[CLS]", 1)  # Assuming [CLS] is the start token

    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        dec_outputs, h, c = decoder_model.predict([empty_target_seq] + stat, verbose=0)

        # Get the index of the sampled word
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_word_index, "[UNK]")  # Convert index back to word

        if sampled_word in ['[SEP]', '[PAD]', '[UNK]'] or len(decoded_translation.split()) > max_sequence_len_decoder:
            stop_condition = True

        if sampled_word not in ['[SEP]', '[PAD]', '[UNK]']:
            decoded_translation += sampled_word + ' '

        # Prepare the next target sequence
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index

        # Update states
        stat = [h, c]

    return decoded_translation.strip()


#-------------OUR------------------#
def generate_answer_our(question):
    
    answer = chatBot(question)
    return answer

#-------------Groq-----------------#
def generate_answer_groq(question):
    prompt = question
    client = Groq(api_key='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')
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    answer = chat_completion.choices[0].message.content
    return answer

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Groq-based chatbot route
@app.route('/chatbot_groq')
def chatbot_groq():
    return render_template('chatbot_groq.html')

# New chatbot route
@app.route('/chatbot_new')
def chatbot_new():
    return render_template('chatbot_our.html')

# Groq answer generation
@app.route('/groq_answer', methods=['POST'])
def groq_answer():
    question = request.json['question']
    answer = generate_answer_groq(question)
    
    result = text_analytics_clint.analyze_sentiment(documents=[question], show_opinion_mining=True)
    doc = [res for res in result if not res.is_error]
    
    return jsonify({'answer': answer, 'sentiment': doc[0].sentiment})

# New chatbot answer generation
@app.route('/new_answer', methods=['POST'])
def new_answer():
    # Add logic to handle the new chatbot's model here
    question = request.json['question']
    answer = generate_answer_our(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
