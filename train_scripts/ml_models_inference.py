import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
import torch

# === Setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load semantic model ===
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


def run_inference(model_path, vec_path, label_map, books_df, topic):
    """
    Runs semantic search on book descriptions based on a topic, then predicts labels using a trained ML model.

    Returns the top 30 most relevant books with predicted labels and similarity scores.
    """

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    books_df = books_df.dropna(subset=["description"]).copy()

    # Semantic search
    books_df["embedding"] = semantic_model.encode(books_df["description"].tolist()).tolist()
    book_embeddings = torch.tensor(books_df["embedding"].tolist()).to(device)
    topic_embedding = semantic_model.encode(topic, convert_to_tensor=True).to(device)
    similarities = util.cos_sim(topic_embedding, book_embeddings)[0].cpu().numpy()
    books_df["similarity"] = similarities

    # Top 30 similar
    top_books = books_df.sort_values(by="similarity", ascending=False).head(30).copy()

    # Predict
    X = vectorizer.transform(top_books["description"])
    preds = model.predict(X)
    top_books["predicted_label"] = [list(label_map.keys())[list(label_map.values()).index(p)] for p in preds]

    return top_books


if __name__ == "__main__":
    topic = input("Enter a topic (e.g., 'AI'): ")
    mode = input("Choose pairing style: 'beginner/advanced' or 'theory/practice': ").strip().lower()

    label_map_level = {"Beginner": 0, "Advanced": 1}
    label_map_type = {"Theory": 0, "Practice": 1}

    books = pd.read_csv("../data/outputs/all_books.csv")

    if mode == "beginner/advanced":
        print("\n=== LEVEL PAIR RECOMMENDATIONS ===")
        result = run_inference("ml/classical_level_model.pkl", "ml/level_vectorizer.pkl", label_map_level, books, topic)
        beginner_books = result[result["predicted_label"] == "Beginner"].head(3)
        advanced_books = result[result["predicted_label"] == "Advanced"].head(3)
        print("\n--- Beginner Books ---")
        print(beginner_books[["title", "authors", "infoLink"]])
        print("\n--- Advanced Books ---")
        print(advanced_books[["title", "authors", "infoLink"]])

    elif mode == "theory/practice":
        print("\n=== TYPE PAIR RECOMMENDATIONS ===")
        result = run_inference("ml/classical_type_model.pkl", "ml/type_vectorizer.pkl", label_map_type, books, topic)
        theory_books = result[result["predicted_label"] == "Theory"].head(3)
        practice_books = result[result["predicted_label"] == "Practice"].head(3)
        print("\n--- Theory Books ---")
        print(theory_books[["title", "authors", "infoLink"]])
        print("\n--- Practice Books ---")
        print(practice_books[["title", "authors", "infoLink"]])

    else:
        raise ValueError("Invalid mode. Choose 'level' or 'type'.")