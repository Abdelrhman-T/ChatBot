# Chatbot with TensorFlow and Seq2Seq Model

This project implements a chatbot using a Seq2Seq (sequence-to-sequence) model with TensorFlow, utilizing an LSTM-based architecture for text generation. It includes preprocessing using SpaCy, handling time conversions, and tokenization.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

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


4. **Run the chatbot:**

    Run the chatbot script in interactive mode (ensure you have a compatible terminal or environment):

    ```bash
    python .\app.py

5. **Model Training (optional):**
   If you need to retrain the model, run the training notebook found in notebooks/TrainModel.ipynb.
