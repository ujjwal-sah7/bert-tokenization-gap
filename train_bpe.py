from tokenizers import ByteLevelBPETokenizer
from transformers import BertTokenizerFast, Trainer, TrainingArguments
from utils import load_data, compute_metrics, get_model

dataset = load_data()

# Train BPE tokenizer
texts = dataset["train"]["text"][:5000]

with open("temp.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t + "\n")

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files="temp.txt", vocab_size=30000, min_frequency=2)

tokenizer.save_model("bpe_tokenizer")

hf_tokenizer = BertTokenizerFast.from_pretrained("bpe_tokenizer")

def tokenize(example):
    return hf_tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)

model = get_model()

training_args = TrainingArguments(
    output_dir="../models/bpe",
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