
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATASET_PATH = Path("dataset")
categories = [
    "business",
    "entertainment",
    "politics",
    "sport",
    "tech"
]

documents = []
labels = []

for category in categories:
    folder = DATASET_PATH / category

    for file_path in folder.glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(
                encoding="latin-1"
            )

        if text.strip():
            documents.append(text)
            labels.append(category)

print("Total documents:", len(documents))
print("Categories:", categories)

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\nTraining documents:", len(X_train))
print("Testing documents:", len(X_test))

# ==========================================
# 3. TF-IDF FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF feature extraction completed!")
print("Number of features:", X_train_tfidf.shape[1])

# ==========================================
# 4. DEFINE FOUR CLASSIFIERS
# ==========================================

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute"
    ),

    "SVM": LinearSVC(
        C=1.0,
        random_state=42,
        max_iter=5000
    ),

    "Naive Bayes": MultinomialNB()
}

results = []
predictions = {}

# Create output directory
os.makedirs("results", exist_ok=True)

# ==========================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        zero_division=0
    ))

# ==========================================
# 6. COMPARATIVE RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "results/comparison.csv",
    index=False
)

# ==========================================
# 7. PERFORMANCE COMPARISON GRAPH
# ==========================================

plot_df = results_df.set_index("Algorithm")

plot_df.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05)
)

plt.title("Comparison of Four Text Classification Algorithms")
plt.ylabel("Performance Score")
plt.xlabel("Classification Algorithm")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("results/performance_comparison.png")
plt.show()

# ==========================================
# 8. CONFUSION MATRICES
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for ax, (name, y_pred) in zip(
    axes.flatten(), predictions.items()
):
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=categories
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")

plt.suptitle("Confusion Matrix Comparison")
plt.tight_layout()

plt.savefig("results/confusion_matrices.png")
plt.show()

# ==========================================
# 9. WHAT-IF EXPERIMENT: CHANGE SVM C
# ==========================================

print("\n========== WHAT-IF EXPERIMENT ==========")

svm_values = [0.1, 1.0, 10.0]
what_if_results = []

for c_value in svm_values:

    experiment_model = LinearSVC(
        C=c_value,
        random_state=42,
        max_iter=5000
    )

    experiment_model.fit(
        X_train_tfidf,
        y_train
    )

    experiment_pred = experiment_model.predict(
        X_test_tfidf
    )

    score = accuracy_score(
        y_test,
        experiment_pred
    )

    what_if_results.append({
        "SVM C Value": c_value,
        "Accuracy": score
    })

    print(
        f"C = {c_value}: Accuracy = {score:.4f}"
    )

what_if_df = pd.DataFrame(what_if_results)

what_if_df.to_csv(
    "results/what_if_experiment.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.plot(
    what_if_df["SVM C Value"],
    what_if_df["Accuracy"],
    marker="o"
)

plt.title("What-If Experiment: Effect of SVM C")
plt.xlabel("SVM C Parameter")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()

plt.savefig("results/what_if_experiment.png")
plt.show()

# ==========================================
# 10. BEST PERFORMING MODEL
# ==========================================

best_model = results_df.loc[
    results_df["F1-Score"].idxmax()
]

print("\n========== FINAL RECOMMENDATION ==========")

print("Best Algorithm:", best_model["Algorithm"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("Precision:", round(best_model["Precision"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("F1-Score:", round(best_model["F1-Score"], 4))

print("\nAll experiments completed successfully!")
print("Check the results folder for graphs and CSV files.")
