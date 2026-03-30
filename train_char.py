from transformers import BertTokenizerFast, Trainer, TrainingArguments
from utils import load_data, compute_metrics, get_model

dataset = load_data()

# Build character vocab
chars = set()
for text in dataset["train"]["text"][:5000]:
    chars.update(list(text))

vocab = {c: i+5 for i, c in enumerate(chars)}
vocab["[PAD]"] = 0
vocab["[UNK]"] = 1
vocab["[CLS]"] = 2
vocab["[SEP]"] = 3
vocab["[MASK]"] = 4

# Save vocab
with open("char_vocab.txt", "w") as f:
    for k in vocab:
        f.write(k + "\n")

tokenizer = BertTokenizerFast(vocab_file="char_vocab.txt")

def tokenize(example):
    return tokenizer(list(example["text"]), truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)

model = get_model()

training_args = TrainingArguments(
    output_dir="../models/char",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=dataset["test"].select(range(1000)),
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()