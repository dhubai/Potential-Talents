# Potential Talents

An intelligent candidate ranking system that helps talent sourcing teams find the best-fit candidates for any role — and gets smarter with every piece of human feedback.

## The Problem

Talent sourcing companies search for candidates using keywords like *"aspiring human resources"* or *"full-stack software engineer"*. This returns a long list of people, but:
- The best candidate might not be at the top.
- A human reviewer has to manually inspect every candidate.
- When the reviewer finds a great match (say, the 7th candidate), there's no way to tell the system *"find me more people like this one."*

This project solves all three problems.

## What This Project Does

1. **Ranks candidates automatically** — Given a role query, the system scores every candidate from 0 (poor match) to 1 (strong match) based on how well their job title matches the query.

2. **Learns from reviewer feedback** — When a reviewer "stars" a candidate they like (or "skips" one they don't), the system adjusts its understanding of what the reviewer is looking for and re-ranks the entire list. Each starring action makes the ranking better.

3. **Filters out irrelevant candidates** — Candidates who clearly don't belong (e.g. a biology student in an HR search) are automatically flagged and can be removed.

4. **Works for any role** — Change the search keywords, and the system adapts. No retraining needed.

5. **Reduces bias** — Location is deliberately excluded from scoring so candidates aren't penalized for where they live. The initial ranking is purely algorithmic with no human preferences baked in.

## How It Works (In Plain English)

- The system reads each candidate's job title and converts it into a numerical representation (an "embedding") that captures its meaning — not just the exact words, but the intent behind them.
- It does the same for the search query.
- Candidates whose meaning is closest to the query get the highest scores.
- When a reviewer stars a candidate, the system shifts its understanding of the query *toward* that person's profile. When a candidate is skipped, it shifts *away*. This is called **relevance feedback**.
- The result: after just 1-2 rounds of feedback, the top of the list is filled with exactly the kind of person the reviewer wants.

## How It Works (Technical Detail)

| Component | Method |
|-----------|--------|
| Text representation | Sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dimensions) |
| Similarity | Cosine similarity between query and candidate embeddings |
| Feature engineering | Embedding similarity (85%) + lexical Jaccard overlap (10%) + connection strength (5%) |
| Score calibration | Sigmoid function maps raw score to [0, 1] |
| Feedback mechanism | Rocchio relevance feedback shifts query vector toward starred / away from skipped |
| Evaluation metrics | NDCG@K and MAP@K — standard ranking quality measures |
| Cutoff policy | Adaptive 70th-percentile threshold with 0.60 floor |
| Hyperparameter tuning | Grid search over feature weights and Rocchio parameters |

## Results

| Metric | Before Feedback | After 2 Rounds of Feedback |
|--------|-----------------|---------------------------|
| NDCG@10 | 0.00 | 0.50 |
| MAP@10 | 0.00 | 0.33 |

The ranking quality improves significantly with each round of human feedback, proving the system learns effectively from reviewer input.

## Project Structure

```
PotentialTalents.ipynb          # Main notebook — full walkthrough with charts
potential-talents.xlsx           # Candidate dataset (104 candidates)
requirements.txt                 # Python dependencies
pyproject.toml                   # Package configuration

talentfit/                       # Core Python package
  ingest.py                      #   Data loading and validation
  text.py                        #   Text cleaning and normalization
  embed.py                       #   Embedding model + disk caching
  retrieve.py                    #   Top-K candidate retrieval
  features.py                    #   Feature engineering and scoring
  ranker.py                      #   Main ranking engine + feedback loop
  eval.py                        #   NDCG and MAP evaluation metrics
  cutoff.py                      #   Adaptive cutoff policy
  feedback.py                    #   Star/skip event storage
  reporting.py                   #   Impact reporting
  legacy_baseline.py             #   Original BERT + Doc2Vec approach (for comparison)

scripts/                         # Command-line tools
tests/                           # Automated tests (pytest)
notebooks/                       # Extra notebook copy
```

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m spacy download en_core_web_sm
```

## Quick Start

### Run the notebook (recommended)
Open `PotentialTalents.ipynb` and run all cells. The notebook walks through:
- Data exploration
- Initial ranking
- Interactive feedback (you choose who to star/skip)
- Evaluation charts showing improvement
- Hyperparameter analysis
- Cutoff and filtering

### Run from the command line

**Rank candidates:**
```bash
python scripts/run_ranker_v1.py --query "aspiring human resources" --query "seeking human resources" --topk 10
```

**Add feedback and rerank:**
```bash
python scripts/feedback_demo.py --query "aspiring human resources" --query "seeking human resources" --star 99 --skip 6 49 --topk 10
```

**Evaluate improvement:**
```bash
python scripts/evaluation_demo.py --query "aspiring human resources" --query "seeking human resources" --ideal 53 --star 53 --k 10
```

## Design Decisions

- **Location is not used in scoring** — to avoid geographic bias. It's kept as a display/filter field only.
- **No supervised training** — the `fit` column has no labels, so we use embedding similarity + feedback instead of training a classifier.
- **Feedback is mathematical, not subjective** — starring shifts a vector in embedding space (Rocchio formula), preventing arbitrary manual manipulation.
- **The cutoff adapts automatically** — a percentile-based threshold means the system works for any role query without manual tuning.

## Technologies Used

- Python 3.12
- Sentence-Transformers (all-MiniLM-L6-v2)
- scikit-learn, pandas, NumPy
- spaCy, NLTK
- Matplotlib
- pytest
