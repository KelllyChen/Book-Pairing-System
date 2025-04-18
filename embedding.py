import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Read the data
def generate_embeddings(input_csv, output_csv, model_name="all-MiniLM-L6-v2"):
    df = pd.read_csv(input_csv).dropna(subset=["description"]).reset_index(drop=True)
    model = SentenceTransformer(model_name)
    df['embedding'] = model.encode(df["description"].tolist(), show_progress_bar=True).tolist()
    df.to_csv(output_csv, index=False)
    print("Embedding saved to file")

if __name__ == "__main__":
    generate_embeddings("data/labeled_books.csv", "data/books_with_embeddings.csv")