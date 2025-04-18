import streamlit as st
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification

# Load models once
@st.cache_resource
def load_models():
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    level_model = BertForSequenceClassification.from_pretrained("bert_level_classifier").to(device)
    type_model = BertForSequenceClassification.from_pretrained("bert_type_classifier").to(device)
    tokenizer_level = BertTokenizer.from_pretrained("bert_level_classifier")
    tokenizer_type = BertTokenizer.from_pretrained("bert_type_classifier")
    return semantic_model, level_model, type_model, tokenizer_level, tokenizer_type

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load book data
df = pd.read_csv("data/books_with_embeddings.csv").dropna(subset=["embedding", "description"])
df["embedding"] = df["embedding"].apply(eval)
book_embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")

# App UI
st.title("📚 Paired Book Recommendation System")
topic = st.text_input("Enter a topic (e.g., NLP, AI ethics, transformers)")
style = st.selectbox("Choose pairing style", ["Beginner ➜ Advanced", "Theory ➜ Practice"])

if topic:
    with st.spinner("Finding books..."):
        semantic_model, level_model, type_model, tokenizer_level, tokenizer_type = load_models()

        topic_embedding = semantic_model.encode(topic, convert_to_tensor=True).to(device)
        similarities = util.cos_sim(topic_embedding, torch.tensor(book_embeddings).to(device))[0].cpu().numpy()
        df["similarity"] = similarities
        top_books = df.sort_values(by="similarity", ascending=False).head(50).copy()

        if style == "Beginner ➜ Advanced":
            model = level_model
            tokenizer = tokenizer_level
            label_map = {0: "Beginner", 1: "Advanced"}
        else:
            model = type_model
            tokenizer = tokenizer_type
            label_map = {0: "Theory", 1: "Practice"}

        # Predict labels
        def predict(descriptions):
            encodings = tokenizer(descriptions, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**encodings)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            return preds

        top_books["predicted_label"] = predict(top_books["description"].tolist())
        top_books["predicted_label"] = top_books["predicted_label"].map(label_map)

        # Display 3 books per group
        group_a, group_b = list(label_map.values())
        for group in [group_a, group_b]:
            st.subheader(f"📘 Top 3 {group} Books on '{topic}'")
            for _, row in top_books[top_books["predicted_label"] == group].head(3).iterrows():
                st.markdown(f"**{row['title']}** (Similarity: {row['similarity']:.2f})")
                st.write(f"👤 *By:* {row.get('authors', 'Unknown')}")
                st.write(row['description'][:300] + "...")
                st.markdown(f"[🔗 More Info]({row.get('infoLink', '#')})")
                st.markdown("---")
