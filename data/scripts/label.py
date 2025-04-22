import pandas as pd
import google.generativeai as genai
import time
import json
import re
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Create Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

PROMPT_TEMPLATE = """
You're labeling AI-related books into two categories:

Level: Beginner or Advanced  
Type: Theory or Practice

Label the following book:

Title: "{title}"
Description: "{description}"

Respond ONLY in valid JSON like this: 
{{"level": "Beginner", "type": "Theory"}}
Do not include any extra text or formatting.
"""

def label_books(df):
    """
    Use gemini to label all books
    Levels: Beginner/Advanced
    Types: Theory/Practice
    """
    levels = []
    types = []

    for idx, row in df.iterrows():
        title = row["title"]
        description = row["description"]
        prompt = PROMPT_TEMPLATE.format(title=title, description=description)

        print(f"Labeling ({idx+1}/{len(df)}): {title[:50]}...")

        try:
            response = model.generate_content(prompt)
            raw_output = response.text.strip()
            cleaned_output = re.sub(r"^```(?:json)?|```$", "", raw_output, flags=re.IGNORECASE).strip()
            label = json.loads(cleaned_output)

            level = label.get("level", "Uncertain")
            typ = label.get("type", "Uncertain")

        except Exception as e:
            print(f"Error on row {idx}: {e}")
            print(f"Raw output: {raw_output}")
            level, typ = "Uncertain", "Uncertain"

        levels.append(level)
        types.append(typ)
        time.sleep(1.2)

    return levels, types

if __name__ == "__main__":
    df = pd.read_csv("data/all_books.csv").dropna(subset=["description"]).reset_index(drop=True)
    levels, types = label_books(df)
    df["level"] = levels
    df["type"] = types
    df.to_csv("../outputs/labeled_books.csv", index=False)
    print("Saved to 'labeled_books.csv'")
