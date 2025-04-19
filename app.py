import streamlit as st
import pandas as pd
import torch
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig
from safetensors.torch import load_file

from dotenv import load_dotenv
import os

load_dotenv()  # loads .env contents into environment
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

# === Setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# === Load models once ===
@st.cache_resource
def load_models():
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    level_model = BertForSequenceClassification.from_pretrained("bert_level_classifier").to(device)
    type_model = BertForSequenceClassification.from_pretrained("bert_type_classifier").to(device)
    tokenizer_level = BertTokenizer.from_pretrained("bert_level_classifier")
    tokenizer_type = BertTokenizer.from_pretrained("bert_type_classifier")
    return semantic_model, level_model, type_model, tokenizer_level, tokenizer_type

# === Google Books API search ===
def fetch_books_from_google(query, max_results=30):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": min(max_results, 40),
        "printType": "books",
        "langRestrict": "en",
        "key": GOOGLE_BOOKS_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error("❌ Failed to fetch books from Google Books API.")
        return pd.DataFrame()

    books = []
    for item in response.json().get("items", []):
        info = item.get("volumeInfo", {})
        if "description" not in info:
            continue
        books.append({
            "title": info.get("title", "Unknown Title"),
            "description": info.get("description", ""),
            "authors": ", ".join(info.get("authors", [])) if info.get("authors") else "Unknown",
            "infoLink": info.get("infoLink", "#")
        })
    return pd.DataFrame(books)

# === Streamlit UI ===
st.title("Paired Book Recommendation System")

topic = st.text_input("Enter a topic (e.g., NLP, AI, ethics):")
style = st.radio("Choose pairing style:", ["Beginner \u2794 Advanced", "Theory \u2794 Practice"])

if topic:
    with st.spinner("Searching and analyzing books..."):

        # Step 1: Fetch books
        df = fetch_books_from_google(topic)
        if df.empty:
            st.warning("No books found.")
            st.stop()

        # Step 2: Load models and embed
        semantic_model, level_model, type_model, tokenizer_level, tokenizer_type = load_models()
        df["embedding"] = semantic_model.encode(df["description"].tolist()).tolist()
        book_embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")

        # Step 3: Classifier setup
        if style == "Beginner \u2794 Advanced":
            model = level_model
            tokenizer = tokenizer_level
            label_map = {0: "Beginner", 1: "Advanced"}
        else:
            model = type_model
            tokenizer = tokenizer_type
            label_map = {0: "Theory", 1: "Practice"}

        # Step 4: Semantic similarity
        topic_embedding = semantic_model.encode(topic, convert_to_tensor=True).to(device)
        similarities = util.cos_sim(topic_embedding, torch.tensor(book_embeddings).to(device))[0].cpu().numpy()
        df["similarity"] = similarities
        top_books = df.sort_values(by="similarity", ascending=False).head(30).copy()

        # Step 5: Predict labels
        def predict_labels(descriptions):
            encodings = tokenizer(
                descriptions, truncation=True, padding=True, max_length=256, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**encodings)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            return preds

        top_books["predicted_label"] = predict_labels(top_books["description"].tolist())
        top_books["predicted_label"] = top_books["predicted_label"].map(label_map)

        # Step 6: Display results
        group_a, group_b = list(label_map.values())

        for group in [group_a, group_b]:
            st.subheader(f"Top 3 {group} Books on '{topic}'")
            selected = top_books[top_books["predicted_label"] == group].head(3)
            if selected.empty:
                st.warning(f"No {group} books found.")
                continue
            for _, row in selected.iterrows():
                st.markdown(f"**{row['title']}** _(Similarity: {row['similarity']:.2f})_")
                st.write(f"*By:* {row.get('authors', 'Unknown')}")
                st.write(row['description'][:300] + "...")
                st.markdown(f"[More Info]({row.get('infoLink', '#')})")
                st.markdown("---")
