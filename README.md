# Question Response Evaluation System

This is a Streamlit web application for evaluating responses to questions based on cosine similarity with preprocessed text data. It allows users to input a question and their answer, then compares the similarity of the user's answer with the dataset to provide feedback.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This project aims to demonstrate how to build a simple question-response evaluation system using Python, Streamlit, and scikit-learn. The system preprocesses text data, calculates cosine similarity, and provides feedback on the correctness of the user's answer.

## Features

- Text preprocessing using NLTK (Snowball stemmer and stopwords removal).
- Cosine similarity calculation with TF-IDF vectorization.
- Interactive Streamlit interface for user input and feedback.
- Domain selection sidebar for filtering questions.
- Score tracking for correct answers.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/question-response-evaluation.git
2. Navigate to the project directory:
   ```bash
   cd question-response-evaluation
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

## Usage
1. Ensure you have the necessary dataset in CSV format (replace 'r_DataQR.csv' in the code with the actual path to your CSV file).
2. Run the Streamlit app:
```bash
  streamlit run app.py

3. Use the sidebar to select the domain, input a question, and provide your answer for evaluation.
   
