# News Classification – Comparative Study of Machine Learning Algorithms

## Project Overview
This project compares four machine-learning algorithms for classifying news articles into five categories using TF-IDF feature extraction.

## Dataset
BBC News Dataset – 2,225 news articles.

Categories:
- Business: 510
- Entertainment: 386
- Politics: 417
- Sport: 511
- Technology: 401

## Algorithms Used
1. Decision Tree
2. K-Nearest Neighbours (KNN)
3. Support Vector Machine (SVM)
4. Naive Bayes

## Methodology
1. Load the labelled news dataset.
2. Preprocess text using TF-IDF with English stop-word removal.
3. Split the dataset into 80% training and 20% testing data.
4. Train all four classifiers using the same feature representation and data split.
5. Evaluate using Accuracy, Precision, Recall and F1-score.
6. Compare confusion matrices and performance graphs.
7. Conduct a what-if experiment by changing the SVM C parameter.

## Experimental Results

| Algorithm | Accuracy |
|---|---:|
| Decision Tree | 86.07% |
| KNN | 96.85% |
| SVM | 99.10% |
| Naive Bayes | 98.20% |

SVM achieved the highest accuracy and F1-score in this experiment.

## What-If Experiment

The SVM C parameter was changed:

| C | Accuracy |
|---|---:|
| 0.1 | 98.88% |
| 1.0 | 99.10% |
| 10.0 | 98.43% |

C = 1.0 achieved the best result in this experiment.

## Final Recommendation
SVM is the recommended classifier for this dataset and experimental setup because it achieved the highest accuracy, precision, recall and F1-score.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the BBC dataset inside the `dataset` folder using the five category folders.

Run:

```bash
python main.py
```

## Output
The results folder contains:
- Comparative performance graph
- Confusion matrices
- Performance metrics CSV
- What-if experiment graph
- What-if experiment CSV

## Dataset Source
BBC News Dataset: https://derekgreene.com/bbc/

**Note:** Experimental results are from the actual execution of this project.
