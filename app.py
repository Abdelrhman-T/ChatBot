from flask import Flask, render_template, request, jsonify
from groq import Groq

import numpy as np
import pandas as pd
import re
import spacy
from num2words import num2words
import tensorflow as tf


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras.models import load_model


app = Flask(__name__)

from tensorflow.keras.preprocessing.text import Tokenizer

df_process = pd.read_csv("models/df_processed.csv")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_process["input_texts processed"]+df_process["target_texts processed"])


from spacy.lang.en import stop_words
stop_words = stop_words.STOP_WORDS

not_ = ["not", "no", "nor", "n't", "really", "extremely", "very", "don't", "and", "or"]
QUESTION_WORDS = ['what', 'why', 'how', 'who', 'where', 'when', 'which']
not_remove = not_ + QUESTION_WORDS

for word in not_remove:
  if word in stop_words:
    stop_words.remove(word)

nlp = spacy.load("en_core_web_sm")

max_sequence_len_encoder, max_sequence_len_decoder = 17, 66
total_words = len(tokenizer.word_index) + 1

def convert_time_to_words(time_str):
    time_parts = time_str.split(":")
    hours = num2words(time_parts[0])
    minutes = num2words(time_parts[1])
    time_in_words = f"{hours} {minutes}"
    return time_in_words

def preprocess_answer_text(text):
    text = re.sub(r"&", "and", text)
    text = re.sub(r"@\w+|http[^ ]+|#", '', text)
    text = re.sub(r"[^\x00-\x7F]+", "'", text)
    text = re.sub(r"[^\w\s.,_']", '', text)
    text = re.sub(r"!=\s*", 'not ', text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text

def preprocess_question_text(text):
    text = re.sub(r"&", "and", text)
    text = re.sub(r'\bis\b', '', text)
    text = re.sub(r"\\'", "'", text)
    text = re.sub(r"\\[a-z]","",text)
    text = re.sub(r"[.]{2,}","aaa", text)
    text = re.sub(r"[.]"," ",text)
    text = re.sub(r"[,]"," ",text)
    text = re.sub(r" ' ","'",text)
    text = re.sub(r'\d{1,2}:\d{2}', lambda x: convert_time_to_words(x.group()), text)
    text = re.sub(r"\d+", lambda x: num2words(int(x.group())), text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    doc = nlp(text)
    keep_punctuation = {"(", ")"}

    tokens = [token.lemma_ for token in doc if 
              (token.text in keep_punctuation or not token.is_punct) and 
              (token.text not in stop_words or token.pos_ == "VERB")]
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text)
    return text


seq2seq = load_model("models/chatBotAI_model.h5")



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
    input_text = preprocess_question_text(question)

    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_len_encoder, padding='post')

    input_tensor = tf.convert_to_tensor(input_sequence)

    stat = encoder_model.predict(input_tensor, verbose=0)

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index.get("[CLS]", 1) 

    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        dec_outputs, h, c = decoder_model.predict([empty_target_seq] + stat, verbose=0)

        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_word_index, "[UNK]") 

        if sampled_word in ['[SEP]', '[PAD]', '[UNK]', 'sep'] or len(decoded_translation.split()) > max_sequence_len_decoder:
            stop_condition = True

        if sampled_word not in ['[SEP]', '[PAD]', '[UNK]' , 'sep']:
            decoded_translation += sampled_word + ' '

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index

        stat = [h, c]

    return decoded_translation.strip()

def customize_api(text):
    history = [
        {
            "role": "system",
            "content": f"{text}"
        }
    ]
    return history

#-------------Kayo------------------#
def generate_answer_our(question):
    
    answer = chatBot(question)
    return answer

#-------------Groq-----------------#
def translate(question,conversation_history):
    prompt = question
    conversation_history.append({
        "role": "user",
        "content": prompt
    })
    
    client = Groq(api_key='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')
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=conversation_history,
        temperature=0,
        max_tokens=100,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    answer = completion.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })
    
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



# our chatbot answer generation
@app.route('/en_answer', methods=['POST'])
def en_answer():
    # Add logic to handle the new chatbot's model here
    question = request.json['question']
    
    answer = generate_answer_our(question)

    return jsonify({'answer': answer})

# groq chatbot answer generation
toEglish_history = customize_api("You are english-arabic and arabic-english translator don't give me multi translation, don't answer in any quistion just need you convert all text to Englis if it already English just put all text without any addition.")
toArabic_history = customize_api("Translate all text to Arabic accurately. If you find any incorrect words or punctuation errors, correct them first before translating. Do not provide multiple translations or answer any questions—only translate the text.")
@app.route('/ar_answer', methods=['POST'])
def ar_answer():
    # Add logic to handle the new chatbot's model here
    question = request.json['question']
    en_question =  translate(question,toEglish_history)

    answer = generate_answer_our(en_question)

    ar_answer =  translate(answer,toArabic_history)
    return jsonify({'answer': ar_answer})

if __name__ == '__main__':
    app.run(debug=True)
