CSE39 NLP Assignment 2

Name - Ujjwal kumar Sah 
Roll number -  23053857 
Section - CSE -39
Instructor- Dr. Sambit Paharaj

# BERT Tokenization Gap Project
 ** Overview**

This project explores the **limitations of WordPiece tokenization in BERT** and proposes improvements using alternative tokenization techniques.

We compare:

* Baseline BERT (WordPiece Tokenizer)
* BERT with BPE Tokenizer
* BERT with Character-level Tokenizer

##  Objective

The main goal is to analyze how different tokenization strategies affect the performance of BERT in **sentiment analysis tasks**.

## Dataset

* *IMDB Movie Reviews Dataset**
* Binary classification:

  * Positive
  * Negative
##  Models Used

### 🔹 1. Baseline Model

* BERT (bert-base-uncased)
* Default **WordPiece Tokenizer**
### 🔹 2. Improved Model 1 (BPE)

* BERT + **Byte Pair Encoding (BPE) Tokenizer**
* Better handling of subwords

---

### 🔹 3. Improved Model 2 (Character-Level)

* BERT + **Character Tokenizer**
* Eliminates out-of-vocabulary (OOV) issues

-## ⚙️ Project Structure

```
bert-tokenization-gap/
│
├── train_baseline.py
├── train_bpe.py
├── train_char.py
├── utils.py
├── requirements.txt
└── README.md

## 🔧 Installation

```bash
pip install -r requirements.txt
```
## ▶️ How to Run

### Run Baseline Model

```bash
python train_baseline.py
```

### Run BPE Model

```bash
python train_bpe.py
```

### Run Character Model

```bash
python train_char.py
## 📈 Evaluation Metrics

* Accuracy
* F1 Score
* Training Time
* Model Size

---

##  Results (Example)

| Model         | Tokenizer | Accuracy | F1 Score |
| ------------- | --------- | -------- | -------- |
| Baseline BERT | WordPiece | 0.85     | 0.84     |
| BERT + BPE    | BPE       | 0.87     | 0.86     |
| BERT + Char   | Character | 0.83     | 0.82     |

---

##  Analysis

* **WordPiece**: Standard but limited in handling rare words
* **BPE**: Better subword representation → improved performance
* **Character-level**: No OOV problem but slower and less semantic

##  Conclusion

Replacing WordPiece tokenization with BPE improves model performance, while character-level tokenization ensures robustness but may reduce efficiency.

## Future Work

* Dynamic tokenization methods
* Hybrid tokenization approaches
* Larger datasets and longer training
