# COSC3081/3015 – Advanced Programming for Data Science  
## Assignment 3 – Milestone 1: NLP Web-based Data Application

### Overview
This milestone focuses on **Natural Language Processing (NLP)** for classifying clothing reviews.  
Using a provided dataset of ~19,600 reviews, the goal is to **pre-process text**, **generate feature representations**, and **build classification models** to predict whether a reviewer recommends a clothing item.

### Dataset
Source: Modified version of [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) (provided via Canvas).  
Key columns used:
- `Title` – Review title
- `Review Text` – Detailed review
- `Recommended` – Binary label (`0` = not recommended, `1` = recommended)

---

## Tasks

### **Task 1 – Basic Text Pre-processing** (5 marks)
- Tokenize `Review Text` using regex:  
  ```python
  r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
  ```
- Convert to lowercase.
- Remove:
  - Words with a length of less than 2 characters.
  - Stopwords (provided in `stopwords_en.txt`).
  - Words that appear only once in the entire corpus (based on term frequency).
  - The 20 most frequent words in the corpus (based on document frequency).
- Save the processed review texts to **`processed.csv`**.
- Build a unigram vocabulary from the processed reviews:
  - Save it to **`vocab.txt`**.
  - Format: `word_string:word_integer_index` (index starts at 0).
  - Vocabulary must be sorted in alphabetical order.
- All code for Task 1 must be written in **`task1.ipynb`**.

---

### **Task 2 – Feature Representation** (7 marks)
Generate three types of feature vectors for reviews:
1. **Count Vector** (Bag-of-Words) – Based on `vocab.txt`.
2. **Word Embedding (Unweighted)** – Using a chosen pretrained model (e.g., GloVe, Word2Vec, FastText).
3. **Word Embedding (TF-IDF Weighted)** – Weighted average of word vectors.

Save:
- **`count_vectors.txt`** – Sparse format:
  ```
  #review_index,word_index:frequency,word_index:frequency,...
  ```
- All code for Task 2 must be in **`task2_3.ipynb`**.

---

### **Task 3 – Clothing Review Classification** (8 marks)
Conduct experiments to answer:
1. **Language Model Comparison** – Which feature representation performs best?
2. **Additional Information** – Does adding the `Title` improve accuracy?

Requirements:
- Use at least **three models** (e.g., Logistic Regression, SVM, Random Forest).
- Evaluate with **5-fold cross-validation**.
- Compare:
  - Only `Title`
  - Only `Review Text`
  - Both combined

---

## File Structure
```
.
├── milestone-1/
│   ├── notebooks/           # Jupyter notebooks for milestone 1
│   │   ├── task1.ipynb
│   │   └── task2_3.ipynb
│   │
│   ├── data/                # Raw and processed data files
│   │   ├── stopwords_en.txt
│   │   ├── processed.csv
│   │   ├── vocab.txt
│   │   └── count_vectors.txt
│   │
│   └── docs/                # Documentation and requirements
│       ├── AP4DS_2025B_A3 - Milestone 1-1.pdf
│       └── AP4DS_2025B_Rubric_A3_Milestone I NLP.pdf
│
├── README.md
└── .gitignore
```

---

## Submission
Submit a **zip file named with your student ID** containing:
- `task1.ipynb` & `task2_3.ipynb`
- `.py` exports of both notebooks (exact match to `.ipynb` code)
- `vocab.txt`, `count_vectors.txt`, `processed.csv`

---

## Academic Integrity
Follow RMIT's [Academic Integrity](https://www.rmit.edu.au/students/student-essentials/rights-and-responsibilities/academic-integrity) guidelines.  
Do not plagiarise or share your code.

---
