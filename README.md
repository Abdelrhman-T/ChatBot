# Chatbot with TensorFlow and Seq2Seq Model

This project implements a chatbot using a Seq2Seq (sequence-to-sequence) model with TensorFlow, utilizing an LSTM-based architecture for text generation. It includes preprocessing using SpaCy, handling time conversions, and tokenization.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributions](#contributions)
- [Future Work](#future-work)


## Project Overview

This chatbot project was developed as our graduation project in the Microsoft machine learning engineer track at DEPI. The chatbot is divided into two main components:
- **KayoBot**: Built using Seq2Seq architecture with LSTM, focused on technical inquiries related to web development, machine learning, and deep learning.
- **GroqBot**: Integrated using the Groq API to handle other conversational aspects.

The system processes user queries effectively, offering accurate and relevant responses across different topics, even when faced with noisy or ambiguous questions.

## Features
- **Seq2Seq Model** with LSTM-based encoder-decoder architecture.
- **Preprocessing** of input text including:
  - Tokenization using **TensorFlow**.
  - Model inference using saved pre-trained weights.


## Installation

### Prerequisites
Ensure that you have the following installed:
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10.11

### Steps

1. **Clone the repository:**

   ```bash
    git clone https://github.com/Abdelrhman-T/ChatBot.git


2. **Create and activate the environment:**

    You can create the environment using Anaconda:

    ```bash
    conda create --name chatbot_env python=3.10.11
    conda activate chatbot_env


3. **Install the dependencies:**
    Run the following command to install the required packages listed in requirements.txt:

    ```bash
    pip install -r requirements.txt

3. **Download the SpaCy English model:**
    ```bash
    python -m spacy download en_core_web_sm

3. **Download My Model from drive:**
    [KayBot Model](https://drive.google.com/file/d/1B2WcwuInFF2pX4lrAak7RxeKQSYfJmPn/view?usp=sharing)

4. **Run the chatbot:**

    Run the chatbot script in interactive mode (ensure you have a compatible terminal or environment):

    ```bash
    python .\app.py


## Usage
You can test the chatbot by asking technical questions related to topics such as Web Development, Machine Learning, or Deep Learning. The chatbot will attempt to respond with accurate, context-aware answers.

Example Queries:
"Explain the concept of overfitting in machine learning."
"Tell me about the support vector machines"
The chatbot is designed to handle both short and long questions, even those with some noise.


## Contributions
This is our graduation project in the Microsoft machine learning engineer track at DEPI.
Project contributors:
- **Abdelrhman Tarek**: Data preprocessing and model development using Seq2Seq with LSTM.
- **Hussein Adel**: Integration of Groq API and development of the user interface.
- **Mohamed Ayman Mohamed**: Responsible for data collection and preparation.
- **Mohamed Ayman Mahmoud**: Managed deployment on Azure (temporarily unavailable due to credit exhaustion).


## Future Work
We are working on improving the chatbot's capabilities, including:
Expanding the training dataset for broader technical topics.
Enhancing the model's ability to handle more complex and nuanced queries.
Re-enabling deployment on Azure once additional credits are available.
