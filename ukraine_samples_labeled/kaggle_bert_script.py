# =======================================
# Optimized Full BERT Training Script for Kaggle Kernel
# =======================================

import os
import torch
import pickle
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Configurations
samples_path = "/kaggle/input/your-labeled-samples"  # <- adjust to your Kaggle input directory
save_dir = "/kaggle/working/bert_saved_models"
os.makedirs(save_dir, exist_ok=True)

samples = [
    "labeled_ukraine_sample_1_cleaned.parquet",
    "labeled_ukraine_sample_2_cleaned.parquet",
    "labeled_ukraine_sample_3_cleaned.parquet",
    "labeled_ukraine_sample_4_cleaned.parquet",
]

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for sample_file in samples:
    print(f"📦 Training on {sample_file}")

    # Load and prepare data
    df = pd.read_parquet(os.path.join(samples_path, sample_file))
    df = df.dropna(subset=["text", "sentiment_label"])

    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df["label"] = df["sentiment_label"].map(label_map)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    class TweetDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    # Training arguments
    sample_id = sample_file.replace(".parquet", "")
    output_dir = os.path.join(save_dir, sample_id)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=300,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="steps",
        fp16=True if torch.cuda.is_available() else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=True)

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Save pickle version for easy future loading
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved at {final_model_path}")

print("🎯 ALL MODELS TRAINED + SAVED!")
Sent from my iPhone