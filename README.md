# Topic Discovery in Documents

An Information Retrieval mini project that discovers hidden topics in a collection of text documents, classifies new text into those topics, and ranks documents for a search query.

**Author:** _your name_  
**Course:** Information Retrieval  
**Project:** Mini Project 2, Discover topics within a collection of documents

---

## Features

- Discovers topics automatically from unlabeled documents (unsupervised)
- Shows the top keywords of each topic as bar charts
- Lists which documents belong to which topic
- Predicts the best topic for any new text
- Ranks the top 5 matching documents for a query using cosine similarity
- Lets you paste your own documents and choose the number of topics (2 to 8)

---

## Files

| File | Description |
|---|---|
| `index.html` (`topic-discovery.html`) | Web version. HTML, CSS and JavaScript in one file. Runs fully in the browser |
| `app.py` | Python version built with Streamlit and scikit-learn, using the 20 Newsgroups dataset |
| `README.md` | This file |

---

## How to run

### Web version (no installation)
Double-click `topic-discovery.html` (or `index.html`). It opens in any modern browser and works offline.

### Python version
```bash
pip install streamlit scikit-learn pandas
streamlit run app.py
```
The app opens at `http://localhost:8501`. The first run downloads the 20 Newsgroups dataset.

---

## Method

**Pipeline**

```
Documents -> Preprocessing -> TF-IDF vectors -> NMF topics -> Topic prediction + Cosine-ranked search
```

1. **Preprocessing:** lowercase, keep words with 3 or more letters, remove stopwords.
2. **Vocabulary filtering:** drop words in fewer than 2 documents (min_df) and in more than 80% of documents (max_df).
3. **TF-IDF weighting:**
   `weight = tf x idf`, with `idf = ln((1 + N) / (1 + df)) + 1`. Each vector is L2-normalized.
4. **NMF (Non-negative Matrix Factorization):**
   `X ~ W x H`
   - `X`: documents x words (TF-IDF matrix)
   - `W`: documents x topics (how much each document belongs to each topic)
   - `H`: topics x words (the largest entries are the topic keywords)

   Solved with multiplicative update rules for 150 iterations:
   ```
   H <- H * (W^T X) / (W^T W H)
   W <- W * (X H^T) / (W H H^T)
   ```
5. **Topic prediction for new text:** keep `H` fixed and solve for the topic weights of the text. The largest weight is the best topic.
6. **Retrieval:** the query becomes a TF-IDF vector. Because vectors are normalized, the dot product equals cosine similarity, and the top 5 documents are returned.

---

## Information Retrieval concepts used

- Text preprocessing and tokenization
- Vocabulary / index building
- Vector Space Model
- TF-IDF term weighting
- Cosine similarity
- Ranked retrieval
- Dimensionality reduction and topic modeling (NMF)
- Document clustering by strongest topic

---

## Why NMF

| Method | Note |
|---|---|
| LSA (SVD) | Components contain negative values, so topics are hard to read |
| LDA | Probabilistic, slower, needs tuning, and results can vary between runs |
| NMF on TF-IDF | Fast, non-negative, and gives readable, parts-based topics |

---

## Limitations

- Bag-of-words ignores word order and meaning, so synonyms are treated as different words
- No stemming or lemmatization
- Ranking scans all documents instead of using an inverted index
- No formal evaluation (precision, recall, MAP, topic coherence). Quality is judged by inspection
- The number of topics is chosen by the user
- The built-in sample has only 40 short documents on five themes (space, cricket, health, technology, cooking)

---

## Future work

- Use BM25 instead of plain TF-IDF for ranking
- Add stemming, lemmatization and n-grams
- Compare with LDA and report topic coherence
- Add retrieval evaluation with labeled queries (precision, recall, MAP)
- Build an inverted index for scalable search
- Choose the number of topics automatically

---

## Tech stack

HTML, CSS, JavaScript (web version) and Python, scikit-learn, Streamlit (Python version).
