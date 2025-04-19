# Book Pairing System

# 0. Project Overview
Traditional book recommendation systems suggest individual books based on user preferences, rating, or purchase history. However, they fail to provide meaningful book pairings that can enhance learning, compare perspectives, or provide a structured reading journey. This project aim to build three different approaches: a naive rule-based approach, classical machine learning models, and deep learning-based models to recommend meaningful book pairings. Additionally, it includes a web application for the live demonstration. The system recommends two complementary books based on topic similarity and learning progression, making it ideal for deeper understanding and structured learning.

## Key Features
- **Model Approaches:**
    - **Naive Rule-Based Approach:** A simple keyword-based baseline used for comparison
    - **Non-Deep Learning Models:** A classical machine learning model that doesn't rely on deep learning techniques.
    - **Deep Learning Models:** Fine-tuned BERT models for semantic understanding and classification
- **Real-World Application:** A user-friendly web application that users can input a topic and select a pairing style(e.g., Beginner → Advanced, Theory ↔ Practice). The system will show related books and recommend three books for each category.

## Evaluation Metric:
To evaluate the performance of the classification models used in book pairing, **Accuracy**, **Precision**, **Recall**, **F1Score** were applied.
These metrics help assess how well the models distinguish between book levels (e.g., Beginner vs. Advanced) and types (e.g., Theory vs. Practice).

# 1. Running Instruction
- Create venv `python -m venv .venv`
- Activate venv `source .venv/bin/activate`
- Install packages `pip install -r requirements.txt`
To run the Streamlit demo locally, run `streamlit run app.py`

# 2. Approaches
## Naive Mean Model
As a baseline, I implement a simple keyword-based classification method to label books by level (Beginner vs. Advanced) and type (Theory vs. Practice).
### How it works
- Level Classification
    - If the book description contains words like "beginner", "easy", "introduction", or "basic", it is classified as Beginner.
    - Otherwise, it is labeled as Advanced.
- Type Classification
    - If the description includes terms such as "theory", "conceptual", or "fundamental", it is categorized as Theory.
    - Otherwise, it is labeled as Practice.
### Evaluation
- **Accuracy**:  0.4771
- **Precision**: 0.4128
- **Recall**:    0.6101
- **F1 Score**:  0.4924

## Non-Deep Learning Models
## Deep Learning Models
# Application

## Demo Link
[**Book Pairing System**](https://huggingface.co/spaces/kellly/Book-Pairing-System)

## Run Streamlit app locally

To run the code, run the following command:

```bash
streamlit run app.py
```

Click on the Local URL (http://localhost:8501) to open the Streamlit app in your browser.