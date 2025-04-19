import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

BEGINNER_KEYWORDS = ["beginner", "easy", "intro", "introduction", "basic", "simple", "step-by-step"]
THEORY_KEYWORDS = ["theory", "conceptual", "fundamental", "overview", "principle"]

def classify_by_keywords_level(text):
    text = text.lower()
    if any(keyword in text for keyword in BEGINNER_KEYWORDS):
        return "Beginner"
    return "Advanced"

def classify_by_keywords_type(text):
    text = text.lower()
    if any(keyword in text for keyword in THEORY_KEYWORDS):
        return "Theory"
    return "Practice"

def evaluate_naive_level_model():
    df = pd.read_csv("data/labeled_books.csv").dropna(subset=["description", "level"]).copy()
    df["predicted_label"] = df["description"].apply(classify_by_keywords_level)

    y_true = df["level"]
    y_pred = df["predicted_label"]

    print("=== Evaluation of Keyword Rule-Based Naive Model (Level) ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='Advanced'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, pos_label='Advanced'):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, pos_label='Advanced'):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Beginner", "Advanced"]))

def evaluate_naive_type_model():
    df = pd.read_csv("data/labeled_books.csv").dropna(subset=["description", "type"]).copy()
    df = df[df["type"].isin(["Theory", "Practice"])].copy()  # filter out 'Both'
    df["predicted_label"] = df["description"].apply(classify_by_keywords_type)

    y_true = df["type"]
    y_pred = df["predicted_label"]

    print("=== Evaluation of Keyword Rule-Based Naive Model (Type) ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='binary', pos_label='Practice'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='binary', pos_label='Practice'):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='binary', pos_label='Practice'):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Theory", "Practice"]))

if __name__ == "__main__":
    evaluate_naive_level_model()
    evaluate_naive_type_model()
