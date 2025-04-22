import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
BASE_URL = "https://www.googleapis.com/books/v1/volumes"

SEARCH_QUERIES = [
    "artificial intelligence", "AI", "machine learning", "ML", "deep learning",
    "neural networks", "supervised learning", "unsupervised learning", 
    "natural language processing", "computer vision", "reinforcement learning", 
    "AI ethics", "AI in healthcare", "AI applications", "AI for beginners", 
    "advanced AI", "AI systems", "AI programming", "AI with Python", "intelligent agents",
    "large language models", "recommendation system"
]

def fetch_books(query, max_results=60):
    """
    Use Google Book API to get book data from defined criterias
    """
    books = []
    seen_ids = set()
    for start_index in range(0, max_results, 10):
        params = {
            "q": query,
            "printType": "books",
            "langRestrict": "en",
            "startIndex": start_index,
            "maxResults": 10,
            "key": API_KEY
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data.get("items", []):
                book_id = item.get("id")
                if book_id not in seen_ids:
                    seen_ids.add(book_id)
                    volume_info = item.get("volumeInfo", {})
                    books.append({
                        "title": volume_info.get("title"),
                        "authors": ", ".join(volume_info.get("authors", [])) if volume_info.get("authors") else None,
                        "description": volume_info.get("description"),
                        "categories": ", ".join(volume_info.get("categories", [])) if volume_info.get("categories") else None,
                        "averageRating": volume_info.get("averageRating"),
                        "ratingsCount": volume_info.get("ratingsCount"),
                        "pageCount": volume_info.get("pageCount"),
                        "publishedDate": volume_info.get("publishedDate"),
                        "infoLink": volume_info.get("infoLink"),
                        "query": query
                    })
        else:
            print(f"Error fetching for query: {query} | Status Code: {response.status_code}")
        time.sleep(1)
    return books

if __name__ == "__main__":
    all_books = []
    for query in SEARCH_QUERIES:
        print(f"Fetching books for: {query}")
        books = fetch_books(query)
        all_books.extend(books)

    df = pd.DataFrame(all_books).drop_duplicates(subset=["title", "authors"])
    df.to_csv("../outputs/all_books.csv", index=False)
    print(f"Saved {len(df)} unique books to 'all_books.csv'")
    

