# Candidate Ranking and Sourcing Pipeline

## Overview

This project automates the process of ranking and sourcing candidates using advanced NLP techniques. It leverages embeddings from BERT and Doc2Vec models to evaluate and rank candidates based on job titles, enhancing the efficiency and accuracy of candidate selection.

## Technologies Used

- **Python**: Core programming language for development.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing and operations.
- **SpaCy**: NLP preprocessing and lemmatization.
- **NLTK**: Text processing, including stemming.
- **TensorFlow**: Deep learning framework for BERT models.
- **Transformers**: Hugging Face's library for BERT tokenization and embeddings.
- **Gensim**: Library for training Doc2Vec models.
- **Scikit-learn**: For calculating cosine similarity between embeddings.

## Key Components

### Data Preprocessing

1. **Loading Data**: Read and inspect the dataset to identify missing values and duplicates.
2. **Cleaning Data**: Remove unnecessary characters, replace abbreviations, and apply stemming and lemmatization.

### Embeddings Generation

1. **BERT Embeddings**: Generate contextual embeddings for job titles using the BERT model.
2. **Doc2Vec Embeddings**: Create document vectors for job titles using the Doc2Vec model.

### Similarity Computation

1. **Cosine Similarity**: Compute similarities between query and candidate job titles using both BERT and Doc2Vec embeddings.
2. **Ranking**: Rank candidates based on similarity scores from both models and calculate mean scores.

### Re-Ranking

1. **Starred Candidates**: Allow users to mark candidates as starred and re-rank based on starred candidates' similarity.
2. **Final Ranking**: Generate final rankings considering both original and starred candidate scores.

## Future Improvements

- **Performance Optimization**: Enhance the efficiency of the pipeline.
- **Extended Features**: Integrate additional NLP models or custom scoring algorithms.
- **User Interface**: Develop a web-based interface for easier interaction.
