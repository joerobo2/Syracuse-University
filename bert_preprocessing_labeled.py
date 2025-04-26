# bert_preprocessing_labeling.py

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

# --------------------------------------------
# Step 1: Setup
# --------------------------------------------

# Download VADER if not already
nltk.download('vader_lexicon')

# Paths
DATA_DIR = "/home/ec2-user/Syracuse-University/ukraine_samples"
OUTPUT_DIR = "/home/ec2-user/Syracuse-University/ukraine_samples_labeled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of sample files
sample_files = [
    "ukraine_sample_1_cleaned.parquet",
    "ukraine_sample_2_cleaned.parquet",
    "ukraine_sample_3_cleaned.parquet",
    "ukraine_sample_4_cleaned.parquet",
]

# Initialize VADER and BERT tokenizer
sentiment_analyzer = SentimentIntensityAnalyzer()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --------------------------------------------
# Step 2: Functions
# --------------------------------------------

def get_sentiment_label(text):
    """Use VADER to assign Positive, Neutral, or Negative label."""
    score = sentiment_analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def preprocess_text_for_bert(text):
    """Tokenize text for BERT (minimal cleaning needed)."""
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# --------------------------------------------
# Step 3: Process Each Sample
# --------------------------------------------

for file_name in sample_files:
    print(f"📄 Processing: {file_name}")

    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_parquet(file_path)

    # Drop rows with missing or empty text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    # Auto-label sentiment
    df["sentiment_label"] = df["text"].apply(get_sentiment_label)

    # Optional: preview label distribution
    print(df["sentiment_label"].value_counts(normalize=True))

    # Save labeled file
    output_path = os.path.join(OUTPUT_DIR, f"labeled_{file_name}")
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved: {output_path}\n")

print("🎯 Preprocessing + Labeling complete!")
