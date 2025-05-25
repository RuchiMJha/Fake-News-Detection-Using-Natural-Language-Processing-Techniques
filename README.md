# Fake-News-Detection-Using-Natural-Language-Processing-Techniques
**Team G4 – NYU Tandon & Stern**

This repository contains the implementation of a stance-based fake news detection model developed for the Fake News Challenge (FNC-1). The task involves classifying the stance of a news article body with respect to a given headline into one of four categories: `agree`, `disagree`, `discuss`, or `unrelated`.

## Table of Contents

- [Overview](#overview)  
- [Business Motivation](#business-motivation)  
- [Dataset Description](#dataset-description)  
- [Methodology](#methodology)  
- [Model Submissions](#model-submissions)  
- [Evaluation](#evaluation)  
- [Usage Instructions](#usage-instructions)  
- [Contributors](#contributors)  
- [License](#license)

## Overview

The rise of misinformation presents a significant challenge to information integrity and public trust. Rather than using binary classification, this project employs a stance-based approach that evaluates the semantic consistency between headlines and their corresponding article bodies. The approach provides more nuanced and interpretable outputs suitable for downstream content moderation and early misinformation flagging.

Key techniques include:
- Lexical and semantic feature engineering
- Use of TF-IDF, cosine similarity, and refuting word presence
- Multiple machine learning classifiers including XGBoost and LightGBM

## Business Motivation

Traditional fake news classifiers often rely on external fact-checking and are limited by the availability of verified claims. Our model addresses this limitation by analyzing internal consistency—treating discrepancies between headlines and article bodies as potential indicators of misinformation.

The model is applicable for:
- News publishing platforms
- Social media content moderation
- Fact-checking automation pipelines
- Media observatories and regulatory tools

For more detailed analysis and business impact estimates, refer to the full [Project Report](./G4_Report.pdf).

## Dataset Description

The project is based on the FNC-1 dataset, which includes approximately 50,000 headline-body pairs. The dataset is divided into training and test sets as follows:

| File | Description |
|------|-------------|
| `train_stances.csv` | Labeled training set (headline, body ID, stance) |
| `train_bodies.csv` | Article bodies for training |
| `test_stances_unlabeled.csv` | Unlabeled test set for model inference |
| `test_bodies.csv` | Corresponding article bodies for test set |
| `competition_test_stances.csv` | Ground truth for final evaluation |
| `competition_test_stances_unlabeled.csv` | Test inputs for final competition |
| `competition_test_bodies.csv` | Final test article bodies |

## Methodology

We approached the problem as a multi-class classification task using a combination of feature engineering and machine learning models.

Feature categories:
- **Lexical Features**: TF-IDF overlap, n-gram overlap, refuting word count
- **Semantic Features**: GloVe embeddings, cosine similarity
- **Sentiment Features**: Polarity and subjectivity differences (TextBlob)
- **Structural Features**: Headline/body length and ratio

Modeling was conducted in the Jupyter notebook [`news_classification.ipynb`](./news_classification.ipynb).

## Model Submissions

The following submissions contain predictions for the FNC-1 test set:

| Model | Submission File |
|-------|------------------|
| Logistic Regression | `submission_lr.csv` |
| Random Forest | `submission_rf.csv` |
| k-Nearest Neighbors | `submission_knn.csv` |
| LightGBM | `submission_lgb.csv` |
| XGBoost | `submission_xgb.csv` |

Each file contains:
```
Headline, Body ID, Stance
```

## Evaluation

Evaluation was performed using the official scoring script `scorer.py`, which uses a weighted accuracy metric:
- +0.25 for correctly classifying related/unrelated
- +0.75 for correctly predicting the specific stance (agree, disagree, discuss)

### Example usage:
```bash
python scorer.py competition_test_stances.csv submission_xgb.csv
```

Model highlights:
- XGBoost and LightGBM achieved top accuracy (~89%)
- Confusion matrices and ROC curves available in the report
- KNN was least effective, confirming its limitations for high-dimensional NLP

## Usage Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repository>.git
   cd <your-repository>
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Jupyter notebook:
   ```bash
   jupyter notebook news_classification.ipynb
   ```

4. Evaluate a prediction file:
   ```bash
   python scorer.py competition_test_stances.csv submission_xgb.csv
   ```

## Contributors

This project was developed as part of the course *Data Science for Business: Technical* (Spring 2025) at New York University.

**Team G4:**
- Ruchi Jha 
- Asmita Sonavane  
- Tanvi Takavane   
- Srushti Shah   
**Faculty Advisor:** Prof. Chris Volinsky

## License

This repository is licensed under the MIT License.  
See [`LICENSE.txt`](./LICENSE.txt) for details.
