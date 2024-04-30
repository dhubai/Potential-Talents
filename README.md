# Talent Sourcing and Ranking System

## Overview
This project aims to automate and optimize the process of talent sourcing and ranking for technology companies. By leveraging natural language processing (NLP) techniques and machine learning models, the system assists in identifying and prioritizing candidates based on their fit for specific roles.

## Features
- **Automated Data Cleaning**: The system automatically cleans the dataset by removing unnecessary words, replacing abbreviations, and standardizing job titles.
- **Semantic Similarity Calculation**: Utilizing BERT embeddings, the system calculates the semantic similarity between candidate job titles and predefined queries.
- **Candidate Ranking**: Candidates are ranked based on their semantic similarity scores, allowing for efficient candidate selection.
- **Manual Intervention**: Users can manually star or favorite candidates, influencing their ranking in the selection process.
- **Re-ranking**: Upon starring candidates, the system re-ranks the candidates to reflect the manual intervention.

## Setup
1. **Dependencies**: Ensure that all required libraries and dependencies are installed. This includes pandas, numpy, spacy, nltk, transformers, and TensorFlow.
   
2. **Data Loading**: The system loads the candidate data from a Google Sheets link specified in the code. Ensure that the link is accessible and contains the necessary data in CSV format.
   
3. **Execution**: Run the main script to execute the talent sourcing and ranking system. Follow the prompts to star candidates and observe the re-ranking process.

## Usage
1. **Data Preparation**: Ensure that the candidate dataset is structured correctly with relevant columns such as job title and ID.
   
2. **Keyword Specification**: Define the keywords or queries relevant to the roles you are sourcing for. These queries will be used to search for matching candidates.
   
3. **Execution**: Run the script and input the IDs of candidates you want to star when prompted. Observe the ranked list of candidates based on their fit for the specified roles.
