Sports vs Politics Text Classification

Natural Language Understanding – Assignment 1

1. Problem Definition

The objective of this project is to build and compare multiple machine learning models for binary text classification.

Given a news article, the system must automatically classify it into one of two categories:

Politics

Sport

The task evaluates the effectiveness of different feature representations and classification algorithms in distinguishing between domain-specific news content.

The goal is not only to achieve high performance, but also to compare modeling approaches systematically and analyze their behavior.

2. Data Source & Labeling
2.1 Dataset

The dataset used is the BBC News Dataset, containing professionally written news articles.

We specifically used:

Politics articles

Sport articles

Total samples: ~928 documents

Politics ≈ 417

Sport ≈ 511

2.2 Labeling

Each document is labeled based on its folder category:

Files inside politics/ → label: Politics

Files inside sport/ → label: Sport

Labels were normalized to ensure consistent formatting.

3. Preprocessing

The preprocessing pipeline was intentionally kept minimal to preserve semantic richness while ensuring clean numerical representations.

3.1 Text Cleaning

Lowercasing (handled automatically by vectorizers)

Stopword removal

No stemming or lemmatization (to preserve interpretability)

3.2 Train-Test Split

Stratified split

80% training, 20% testing

Fixed random seed for reproducibility

Processed data stored as:

data/processed/
    train.csv
    test.csv
    complete_dataset.csv

4. Dataset Analysis
4.1 Class Distribution

The dataset is relatively balanced:

Politics: ~45%

Sport: ~55%

This reduces risk of bias toward one class.

4.2 Separability

The dataset is highly separable due to domain-specific vocabulary:

Politics examples:

minister

government

parliament

election

Sport examples:

match

goal

coach

cup

This vocabulary distinction contributes to high classification performance.

5. Feature Engineering

Three feature representations were evaluated:

5.1 Bag of Words (BoW)

CountVectorizer

Unigrams only

max_features = 12000

5.2 TF-IDF

TfidfVectorizer

Unigrams

Sublinear term frequency scaling

5.3 N-grams

TfidfVectorizer

Unigrams + Bigrams

min_df = 2

max_features = 18000

N-grams allow capturing contextual phrases such as:

“prime minister”

“world cup”

6. Model Comparison

Three machine learning algorithms were evaluated:

6.1 Multinomial Naive Bayes

Suitable for frequency-based text features

Strong baseline for NLP tasks

6.2 Logistic Regression

Linear classifier

Handles high-dimensional sparse data effectively

6.3 Linear Support Vector Machine

Maximizes decision margin

Strong performance in text classification

Total experiments conducted:

3 models × 3 feature types = 9 experiments

7. Evaluation Protocol
7.1 Cross-Validation

3-fold Stratified Cross-Validation

Hyperparameter tuning using GridSearchCV

Scoring metric: Macro F1

7.2 Test Evaluation

Final evaluation performed on held-out test set using:

Accuracy

Precision (macro & weighted)

Recall (macro & weighted)

F1-score (macro & weighted)

Confusion matrix

7.3 Full Dataset Retraining

After selecting best hyperparameters, models were retrained on the full dataset for final model storage.

8. Results & Analysis
8.1 Overall Performance

Most models achieved near-perfect performance.

This is expected due to:

Strong domain separability

Distinct vocabulary usage

Professional journalistic writing

8.2 Feature Comparison

Observations:

BoW already achieves very high performance.

TF-IDF offers slight robustness but minimal improvement.

N-grams do not significantly outperform unigrams due to already strong separability.

8.3 Model Comparison

Naive Bayes performs extremely well due to conditional independence assumption working effectively in this domain.

Logistic Regression and Linear SVM show nearly identical performance.

All models perform well in sparse high-dimensional feature space.

8.4 Training Time vs Performance

Linear models show slightly higher computational cost than Naive Bayes but achieve comparable performance.

9. Discussion

The dataset’s strong vocabulary separation leads to near-linear separability of classes.

Key insights:

Simpler models perform remarkably well.

Feature engineering improvements yield diminishing returns.

Linear classifiers are sufficient for this binary domain classification.

This confirms that classical ML approaches remain highly effective for structured news classification tasks.

10. Limitations

Despite strong performance, the project has several limitations:

Dataset is highly separable; real-world classification may be harder.

Only binary classification considered.

No noisy or informal text data included.

No deep learning models evaluated.

Temporal or stylistic variation not analyzed.

Future improvements could include:

Multi-class classification

Domain transfer experiments

Testing on noisy user-generated content

11. Reproducibility

The project is fully reproducible:

Fixed random seed

Processed dataset stored as CSV

Cross-validation used

All results saved as CSV files

Trained models stored as .pkl files

Visualization scripts included

Folder structure ensures clean separation between:

Raw data

Processed data

Models

Results

Plots

12. Conclusion

This project demonstrates that classical machine learning models combined with effective feature engineering can achieve near-perfect performance on domain-separable text classification tasks.

Key takeaways:

Bag-of-Words remains a strong baseline.

TF-IDF provides robustness but limited improvement here.

Linear SVM and Logistic Regression are highly competitive.

Naive Bayes remains efficient and effective.

The study confirms that for structured, professionally written domain-specific news data, linear models with sparse representations are sufficient for high-accuracy classification.