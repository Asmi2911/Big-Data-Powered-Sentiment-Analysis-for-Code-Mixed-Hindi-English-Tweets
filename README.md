# Big-Data-Powered-Sentiment-Analysis-for-Code-Mixed-Hindi-English-Tweets

This project presents an end-to-end scalable big data pipeline that performs sentiment classification and trend detection on Hindi-English (Hinglish) code-mixed tweets. Using Apache Spark, HBase, Hive, and a fine-tuned IndicBERT model, the system handles linguistic complexity—such as transliteration, spelling variation, and code-switching—with high accuracy and scalability.

## Key Highlights

- Handles noisy, informal, multilingual social media text at scale.
- Incorporates custom negation and POS-based features for deeper sentiment context.
- Integrates Spark NLP, Ekphrasis, and PySpark MLlib for robust preprocessing and feature engineering.
- Evaluated with IndicGLUE benchmark, achieving over **81% accuracy** and **81% F1 score**.

## Objectives

- Build a scalable sentiment analysis pipeline for Hinglish tweets.
- Handle real-world code-mixed complexity using Big Data tools.
- Apply fine-tuned LLMs (IndicBERT) for sentiment classification.
- Enable semantic search and trend extraction via BM25 indexing.
- Provide future-ready output via Hive for real-time dashboard integration.

## Technologies Used

| Layer               | Tools & Frameworks                                           |
|---------------------|--------------------------------------------------------------|
| Data Storage        | Apache HBase, Hadoop HDFS                                    |
| Text Preprocessing  | Spark NLP, Ekphrasis, Custom Hinglish Normalization          |
| Feature Engineering | PySpark MLlib, TF-IDF, POS Tagging, Negation Detection       |
| Modeling            | IndicBERT via Spark NLP, DistilBERT SST-2 for English Labels |
| Indexing            | BM25 (Probabilistic Search Ranking)                          |
| Evaluation          | Accuracy, F1 Score, IndicGLUE Benchmark                      |
| Visualization       | Matplotlib, Seaborn, Excel Charts                            |

## Repository Structure

├── datasets/ # Cleaned and processed tweet datasets

│ └── final_cleaned.csv

├── flow-graphs/ # Visual assets for result interpretation

│ ├── bargraph_predicted_labels.png

│ ├── evaluation_acc_f1.png

│ └── pie_chart_predicted_labels.png

├── model/ # Trained IndicBERT model

├── bigdata_proj.ipynb # Main pipeline notebook

└── README.md


## Project Pipeline

### 1. Data Collection & Storage
- Dataset: [AI4Bharat – Mann Ki Baat](https://huggingface.co/datasets/ai4bharat/Mann-ki-Baat)
- Stored in Apache HBase for structured retrieval.
- HDFS used for distributed data persistence.

### 2. Preprocessing & Normalization
- Spark NLP for tokenization and transliteration.
- Ekphrasis for emoji/URL/elongation normalization.
- Custom Hinglish regex and dictionary UDFs for spelling standardization.
- Null/duplicate entries dropped; language switching normalized.

### 3. Feature Engineering
- TF-IDF via HashingTF and ChiSqSelector (top 100 features).
- POS tagging with `pos_anc` model; NNN token filtering for noisy tweets.
- Semantic features: `stopword_percentage`, `has_negation` flag.

### 4. Modeling & Training
- IndicBERT via Spark NLP (ClassifierDL) on T4 GPU-backed Spark cluster.
- Trained for 50 epochs on 70:30 train-test split.
- Accuracy: **81.20%**, F1 Score: **81.05%**.
- English tweets pre-labeled using DistilBERT (SST-2) for auto-bootstrapping.

### 5. Search & Indexing
- BM25 applied for semantically relevant trend retrieval.
- Enables keyword-agnostic, language-robust tweet ranking.

### 6. Evaluation & Results
- Benchmarked using [IndicGLUE](https://indicnlp.ai4bharat.org/indic-glue/)
- Sentiment distribution visualized using bar/pie charts.
- POS accuracy compared across multiple Spark NLP models.


## Visual Insights

| Chart Type            | Description                                  |
|------------------------|----------------------------------------------|
| Sentiment Pie Chart    | Shows dominance of negative tweets (~66%)    |
| Accuracy/F1 Bar Graph  | Highlights balanced model performance (81%+) |
| Frequent Words Cloud   | Identifies trending Hinglish terms           |
| POS Comparison Chart   | Shows performance of POS taggers             |


## Challenges Solved

- **Romanized Hindi Variation**: Built regex-based dictionary and transliteration pipeline.
- **Negation Handling**: Manual lexicon and pattern detection to catch emotional reversals.
- **GPU Integration**: Shifted from HuggingFace to Spark NLP IndicBERT for seamless GPU support.
- **Outlier Filtering**: Penalized noisy tweets with high `NNN` POS tags.
- **Memory Bottlenecks**: Managed DistilBERT inference through chunked processing and batch control.

## Future Scope

- 🔤 Support for more Indian languages: Tamil, Bengali, Marathi, etc.
- 📱 Expansion to platforms like Facebook, Instagram, YouTube.
- 📊 Real-time sentiment dashboards using Spark Streaming + Hive + Grafana.
- 🧠 Explainability via SHAP/LIME for transparency in predictions.
- 🤖 Semi-supervised labeling to auto-tag Hinglish tweets.

## Run the Code

```bash
# Step 1: Install Required Libraries
pip install spark-nlp==5.5.3 pyspark ekphrasis transformers

# Step 2: Launch the Notebook
jupyter notebook bigdata_proj.ipynb

## References
AI4Bharat Dataset

Spark NLP by John Snow Labs

IndicBERT

Ekphrasis

