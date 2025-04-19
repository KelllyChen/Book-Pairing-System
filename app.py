import streamlit as st
import pandas as pd
import torch
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Page config
st.set_page_config(page_title="Paired Book Recommendation", layout="centered")

# === Custom CSS for layout and styling ===
st.markdown("""
<style>
body {
    background-color: #f5f7fa !important;
}
[data-testid="stAppViewContainer"] > .main {
    display: flex;
    justify-content: center;
    padding-top: 3rem;
}

.book-card {
    background-color: #f9fafb;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.book-title {
    font-weight: 600;
    font-size: 1.15rem;
    margin-bottom: 0.5rem;
}
.more-info {
    color: #4f46e5;
    text-decoration: underline;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    margin-top: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# === Load models once ===
@st.cache_resource
def load_models():
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    level_model = BertForSequenceClassification.from_pretrained("bert_level_classifier").to(device)
    type_model = BertForSequenceClassification.from_pretrained("bert_type_classifier").to(device)
    tokenizer_level = BertTokenizer.from_pretrained("bert_level_classifier")
    tokenizer_type = BertTokenizer.from_pretrained("bert_type_classifier")
    return semantic_model, level_model, type_model, tokenizer_level, tokenizer_type

# === Fetch books ===
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

# === Predict book categories ===
def predict_labels(descriptions, model, tokenizer):
    encodings = tokenizer(
        descriptions, truncation=True, padding=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return preds

# === Main UI block ===
with st.container():
    st.markdown('<div class="central-box">', unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;'>📚 Paired Book Recommendation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.1rem;'>Enter a topic (like <i>AI</i>, <i>NLP</i>, or <i>ethics</i>) and choose a pairing style. Our system will recommend a pair of books: one for beginners and one for advanced readers, or theory + practice.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 3, 1])
    with col1:
        topic = st.text_input("Topic", placeholder="AI responsible")
    with col2:
        style = st.selectbox("Pairing Style", ["Beginner → Advanced", "Theory → Practice"])
    with col3:
        st.markdown("""
            <style>
            div.stButton > button {
                padding: 0.4rem 1.2rem;
                font-size: 1rem;
                border-radius: 8px;
                white-space: nowrap;
            }
            </style>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0rem'></div>", unsafe_allow_html=True)

        recommend = st.button("Recommend")

    if topic and recommend:
        with st.spinner("🔍 Searching and analyzing books..."):
            df = fetch_books_from_google(topic)
            if df.empty:
                st.warning("No books found.")
                st.stop()

            semantic_model, level_model, type_model, tokenizer_level, tokenizer_type = load_models()

            df["embedding"] = semantic_model.encode(df["description"].tolist()).tolist()
            book_embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")

            topic_embedding = semantic_model.encode(topic, convert_to_tensor=True).to(device)
            similarities = util.cos_sim(topic_embedding, torch.tensor(book_embeddings).to(device))[0].cpu().numpy()
            df["similarity"] = similarities
            top_books = df.sort_values(by="similarity", ascending=False).head(30).copy()

            if style == "Beginner → Advanced":
                model = level_model
                tokenizer = tokenizer_level
                label_map = {0: "Beginner", 1: "Advanced"}
            else:
                model = type_model
                tokenizer = tokenizer_type
                label_map = {0: "Theory", 1: "Practice"}

            top_books["predicted_label"] = predict_labels(top_books["description"].tolist(), model, tokenizer)
            top_books["predicted_label"] = top_books["predicted_label"].map(label_map)

            for group in label_map.values():
                st.markdown(f"<h3 style='margin-top:2rem;'>{group} Books</h3>", unsafe_allow_html=True)
                selected = top_books[top_books["predicted_label"] == group].head(3)
                if selected.empty:
                    st.warning(f"No {group} books found.")
                    continue
                for _, row in selected.iterrows():
                    st.markdown(f"""
                    <div class="book-card">
                        <div class="book-title">{row['title']}</div>
                        <div style="color: #555; font-size: 0.95rem; margin-bottom: 0.4rem;"><i>By {row['authors']}</i></div>
                        <div>{row['description'][:300]}...</div>
                        <a class="more-info" href="{row['infoLink']}" target="_blank">📖 More Info</a>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
