from transformers import BertTokenizer, Trainer, TrainingArguments
from utils import load_data, compute_metrics, get_model

dataset = load_data()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)

model = get_model()

training_args = TrainingArguments(
    output_dir="./models_baseline",
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