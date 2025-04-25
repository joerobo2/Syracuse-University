# Ukraine Conflict Tweets: NLP and Time-Series Analysis

## 📄 Project Overview

This project analyzes ~10M tweets related to the Ukraine-Russia conflict. 
We combine natural language processing (NLP) and time-series analysis to:
- Detect potential misinformation
- Track public sentiment over time
- Understand topic evolution

**Dataset Source:** Aggregated tweets collected via public APIs and scraping between 2022-2023.

---

## 🛠 Project Pipeline

### 1. Data Ingestion
- Combined 289 batch files into a master dataset (~10M rows).
- Split into 4 representative samples (~2.75M rows each).

### 2. Preprocessing
- Dropped columns with >70% missing values.
- Converted timestamps to `datetime` format.
- Preserved critical fields: text, hashtags, user info, metadata.

### 3. Exploratory Data Analysis (EDA)
- Univariate distributions (followers, retweets, likes).
- Language distribution analysis.
- Pairplots of key numeric variables.
- Correlation matrix and scatterplots.
- ANOVA testing to ensure sampling balance across splits.

### 4. Assumptions and Limitations
- Timestamp trust assumed.
- Noise inherent to social media data.
- Possible gaps or sparsity in certain time intervals.
- Potential demographic and geographic bias.

---

## 📊 Key Columns for Analysis

| Column | Description |
|:---|:---|
| `tweetcreatedts` | Timestamp (UTC) |
| `text` | Tweet text (for NLP) |
| `language` | Tweet language |
| `retweetcount` | Retweet count |
| `favorite_count` | Likes |
| `hashtags` | Extracted hashtags |
| `is_retweet` | Whether the tweet is a retweet |
| `followers`, `following` | User metrics |

---

## 📈 Planned Analysis

- **NLP:** Sentiment analysis, misinformation detection, topic modeling.
- **Time-Series:** Trend shifts in sentiment, misinformation over time.
- **Granger Causality:** Explore whether misinformation precedes sentiment changes.
- **Stationarity Testing:** Verify assumptions for time-series forecasting models.

---

## 📂 Repo Structure

```plaintext
Syracuse-University/
├── ukraine_batches/ (raw batch files)
├── ukraine_master_dataset.parquet (full 10M rows)
├── ukraine_samples/
│   ├── ukraine_sample_1.parquet (and sample 2-4)
│   ├── cleaned_samples/ (after preprocessing)
│   ├── EDA results (csv, png)
├── preprocessing_scripts/
│   ├── combine_batches.py
│   ├── preprocess_samples.py
│   ├── samples_eda.py
├── README.md
