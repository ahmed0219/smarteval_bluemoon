# Question Response Evaluation System

This is a Streamlit web application for evaluating responses to questions based on cosine similarity with preprocessed text data. It allows users to input a question and their answer, then compares the similarity of the user's answer with the dataset to provide feedback.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

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
