import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load your dataset
df = pd.read_csv('r_DataQR.csv')  # Assurez-vous de remplacer 'chemin_vers_votre_dataset.csv' par le chemin réel de votre fichier CSV

# Combine question and answer into a single text
df['Combined_text'] = df['Question'] + ' ' + df['Réponse correcte']

# Initialize NLTK's Snowball stemmer for French and stopwords
stemmer = SnowballStemmer('french')
stop_words = set(stopwords.words('french'))

# Function to preprocess text (remove stopwords and perform stemming)
def preprocess_text(text):
    tokens = text.lower().split()  # Convert to lowercase and split into tokens
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]  # Apply stemming and remove stopwords
    return ' '.join(filtered_tokens)

# Apply text preprocessing to the combined text column
df['Processed_text'] = df['Combined_text'].apply(preprocess_text)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed_text'])
answers = df['Réponse correcte'].values

# Precompute cosine similarities
cos_similarities = cosine_similarity(X)

# Streamlit interface
st.title('Évaluation de la réponse à une question')

# Sidebar for domain selection
domain = st.sidebar.selectbox('Sélectionnez le domaine :', df['Domaine'].unique())

# Filter questions based on selected domain
filtered_df = df[df['Domaine'] == domain]

if not filtered_df.empty:
    # Display question and get user input
    question_idx = np.random.choice(filtered_df.index)  # Select a random question index
    question = filtered_df.loc[question_idx, 'Question']
    user_input = st.text_input("Question :", question)

    # Add input box for user's answer
    user_answer = st.text_input("Réponse de l'utilisateur :")

    # Initialize score
    score = 0

    if st.button('Évaluer la réponse'):
        if user_answer:
            # Preprocess user input
            processed_input = preprocess_text(user_answer)

            # Transform preprocessed user input into a vector
            user_input_vector = vectorizer.transform([processed_input])

            # Calculate cosine similarity between user input and dataset answers using sparse matrix operations
            user_cos_similarities = cosine_similarity(user_input_vector, X, dense_output=False).toarray()[0]

            # Set similarity threshold
            similarity_threshold = 0.6  # Adjust as needed

            # Find the most similar answer using np.argmax for efficiency
            most_similar_idx = np.argmax(user_cos_similarities)
            max_similarity = user_cos_similarities[most_similar_idx]

            if max_similarity >= similarity_threshold:
                most_similar_answer = answers[most_similar_idx]
                st.success(f'Bonne réponse ! La réponse correcte est : {most_similar_answer}')
                score += 1  # Increase score for correct answer
                st.write(f'Score: {score}')  # Display current score
            else:
                st.error('Mauvaise réponse.')
        else:
            st.warning('Veuillez entrer une réponse avant de valider.')
else:
    st.warning('Aucune question trouvée pour ce domaine.')
