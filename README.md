---

# 📚 Classical Information Retrieval System

This project implements a **comprehensive Information Retrieval (IR) system** from scratch, designed to parse, process, and rank documents against user queries. Developed and evaluated using the **Cranfield dataset**, this system serves as a benchmark for exploring classical and advanced IR techniques.

---

## 🚀 Features

* **End-to-End IR Pipeline**: Complete text-processing tools, including:

  * Sentence Segmentation
  * Tokenization
  * Stopword Removal
  * Inflection Reduction (Stemming/Lemmatization)

* **Multiple Retrieval Models**:

  * **Vector Space Model (VSM)**: TF-IDF + Cosine Similarity.
  * **Latent Semantic Analysis (LSA)**: Uses SVD to reveal hidden semantic structure.
  * **Case Retrieval Net (CRN)**: Expands queries using a term-term similarity matrix.

* **Typo Correction**: Corrects spelling errors in user queries using edit distance and Jaccard similarity.

* **Autocomplete**: Suggests query completions based on corpus vocabulary.

* **Modular Architecture**: Clean, reusable, and extendable module structure.

* **Comprehensive Evaluation**: Implements standard IR metrics:

  * Precision\@k
  * Mean Average Precision (MAP)

---

## 🏛️ System Architecture

The system is orchestrated via `main.py` and follows a modular, pipelined structure.

### 1. Preprocessing Pipeline

```text
Input Text
  → Sentence Segmentation
  → Tokenization
  → Stopword Removal
  → Inflection Reduction
  → Processed Tokens
```

### 2. Query Handling

* Same preprocessing pipeline as documents
* **Autocomplete**: Suggests terms as user types
* **Typo Correction**: Suggests corrections post-query

### 3. Retrieval and Ranking

Processed query is passed to a retrieval model:

* VSM
* LSA
* CRN

Returns an **ordered list of documents** ranked by relevance.

### 4. Evaluation

Compares ranked documents against ground truth (`qrels` file):

* Calculates MAP and Precision\@k

---

## 🛠️ Core Modules

| Module    | Description                                                         |
| --------- | ------------------------------------------------------------------- |
| `main.py` | Entry point. Manages CLI, preprocessing, retrieval, and evaluation. |
| `util.py` | Manages imports and downloads required NLTK data.                   |

### Preprocessing

* `sentenceSegmentation.py`: Naive or NLTK-based sentence splitting.
* `tokenization.py`: Naive or Penn TreeBank tokenization.
* `stopwordRemoval.py`: NLTK-based stopword removal.
* `stopwordRemoval_bottom_up.py`: IDF-based stopword detection.
* `inflectionReduction.py`: Uses Porter Stemmer and WordNet Lemmatizer.

### Retrieval Models

* `informationRetrieval.py`: Implements **VSM** with TF-IDF and cosine similarity.
* `LSA.py`: Implements **LSA** using SVD on the term-document matrix.
* `CRN.py`: Implements **Case Retrieval Net** for query expansion.

### Query Correction & Autocomplete

* `Vocabulary.py`: Uses bigrams and Jaccard similarity to suggest corrections.
* `Edit_distance.py`: Implements Levenshtein distance to rank candidates.
* Autocomplete is integrated in the interactive query input loop in `main.py`.

### Evaluation

* `evaluation.py`: Computes:

  * Precision\@k
  * Recall
  * Mean Average Precision (MAP)

---

## ⚙️ Setup and Installation

### ✅ Prerequisites

* Python 3.x
* pip


### 📥 Install Dependencies

The system uses `NLTK`, `NumPy`, `SciPy`, and `scikit-learn`. Use pip to install dependencies:

```bash
pip install nltk numpy scipy scikit-learn
```

The first run will automatically download NLTK models via `util.py`.

---

## 💻 Usage

All functionality is controlled via `main.py`.

### 1. Evaluate on Cranfield Dataset

```bash
python main.py -dataset /path/to/cranfield/
```

* Outputs MAP and other metrics.
* Saves results in the `output/` directory.

### 2. Interactive Custom Query

```bash
python main.py -dataset /path/to/cranfield/ -custom
```

* Prompts user for input.
* Offers **autocomplete** suggestions.
* Applies **typo correction** if needed.
* Returns top 5 most relevant document IDs.

---

## 🧾 Command-Line Arguments

| Argument      | Description                                 |
| ------------- | ------------------------------------------- |
| `-dataset`    | Path to dataset folder (e.g., `cranfield/`) |
| `-out_folder` | Output folder path (default: `output/`)     |
| `-segmenter`  | Sentence segmenter: `naive` or `punkt`      |
| `-tokenizer`  | Tokenizer type: `naive` or `ptb`            |
| `-custom`     | Enables custom query mode                   |

---

## 📁 Example Folder Structure

```
your-repository-name/
│
├── cranfield/                 # Dataset folder
├── output/                    # Output results
├── main.py                    # Main script
├── util.py                    # Utilities
├── evaluation.py              # Evaluation metrics
├── informationRetrieval.py    # VSM model
├── LSA.py                     # LSA model
├── CRN.py                     # CRN model
├── sentenceSegmentation.py    # Sentence segmentation
├── tokenization.py            # Tokenization
├── stopwordRemoval.py         # Stopword removal
├── stopwordRemoval_bottom_up.py
├── inflectionReduction.py     # Stemming and lemmatization
├── Vocabulary.py              # Typo correction
├── Edit_distance.py           # Edit distance calculator
```
