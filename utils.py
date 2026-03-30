from datasets import load_dataset
from transformers import BertForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score

def load_data():
    dataset = load_dataset("imdb")
    return dataset

def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def get_model():
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )