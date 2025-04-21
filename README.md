# Book Pairing System

# 0. Project Overview
Traditional book recommendation systems suggest individual books based on user preferences, rating, or purchase history. However, they fail to provide meaningful book pairings that can enhance learning, compare perspectives, or provide a structured reading journey. This project aim to build three different approaches: a naive rule-based approach, classical machine learning models, and deep learning-based models to recommend meaningful book pairings. Additionally, it includes a web application for the live demonstration. The system recommends two complementary books based on topic similarity and learning progression, making it ideal for deeper understanding and structured learning.

## Key Features
- **Model Approaches:**
    - **Naive Rule-Based Approach:** A simple keyword-based baseline used for comparison
    - **Non-Deep Learning Models:** A classical machine learning model that doesn't rely on deep learning techniques.
    - **Deep Learning Models:** Fine-tuned BERT models for semantic understanding and classification
- **Real-World Application:** A user-friendly web application that users can input a topic and select a pairing style(e.g., Beginner → Advanced, Theory ↔ Practice). The system will show related books and recommend three books for each category.

## Evaluation Metric
To evaluate the performance of the classification models used in book pairing, **Accuracy**, **Precision**, **Recall**, **F1Score** were applied.
These metrics help assess how well the models distinguish between book levels (e.g., Beginner vs. Advanced) and types (e.g., Theory vs. Practice).

# 1. Running Instruction
- Get Google Books API key and Gemini API key and save them in .env
- Create venv `python -m venv .venv`
- Activate venv `source .venv/bin/activate`
- Install packages `pip install -r requirements.txt`
To run the Streamlit demo locally, run `streamlit run app.py`

# 2. Data
To keep this project focused and lightweight as a **Minimum Viable Product (MVP)**, I narrowed the book domain to **AI-related topics only**.
- Books data was fetched using **Google Books API**
- Used Google's **Gemini API** to automatically label each book by:
  - **Level:** Beginner or Advanced
  - **Type:** Theory or Practice

# 3. Approaches
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
### Level Classification (Beginner vs. Advanced)

| Class     | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Beginner  | 0.52      | 0.67   | 0.58     |
| Advanced  | 0.66      | 0.51   | 0.57     | 

**Overall Metrics:**

- **Accuracy:** 0.5785  


---

### Type Classification (Theory vs. Practice)

| Class     | Precision | Recall | F1 Score | 
|-----------|-----------|--------|----------|
| Theory    | 0.41      | 0.61   | 0.49     | 
| Practice  | 0.58      | 0.38   | 0.46     | 

**Overall Metrics:**

- **Accuracy:** 0.4771  


## Non-Deep Learning Models
### How it works
- **Text Representation:** TF-IDF (Term Frequency–Inverse Document Frequency) was applied to book descriptions with a max feature size of 5000.
- **Model Used:** Random Forest Classifier
- **Hyperparameter Tuning:** Performed using GridSearchCV with 5-fold cross-validation.
- **Train-Test Split:** 80/20 stratified split to preserve class distribution.
- **Label Encoding:**
  - Beginner = 0, Advanced = 1
  - Theory = 0, Practice = 1
### Level Classification (Beginner vs. Advanced)

| Class     | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Beginner  | 0.73      | 0.86   | 0.79     |
| Advanced  | 0.77      | 0.59   | 0.67     | 

**Overall Metrics:**

- **Accuracy:** 0.7419  


### Type Classification (Theory vs. Practice)

| Class     | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Theory    | 0.75      | 0.88   | 0.81     |
| Practice  | 0.78      | 0.59   | 0.67     | 

**Overall Metrics:**

- **Accuracy:** 0.7582  


## Deep Learning Models
### How it works
- **Model:** [`BertForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification) with 2 output labels and a dropout rate of 0.3.
- **Tokenizer:** `BertTokenizer` with truncation, padding, and a max token length of 256.
- **Text Input:** Raw book descriptions from a labeled dataset (`Theory`, `Practice`) mapped to numeric labels.
- **Dataset Class:** Custom `BookDataset` class extends PyTorch `Dataset` for use with the Hugging Face `Trainer` API.
- **Data Split:** 80/20 stratified train-test split using `train_test_split`.
- **Training Framework:** Hugging Face's `Trainer` + `TrainingArguments`
- **Training Configuration:**
  - Epochs: 4  
  - Batch Size: 16  
  - Learning Rate: 2e-5  
  - Weight Decay: 0.01  
  - Evaluation + Save steps: Every 500 steps  
  - Logging: Every 10 steps  
  - GPU support enabled if available

### Level Classification (Beginner vs. Advanced)

| Class     | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Beginner  | 0.79      | 0.69   | 0.74     |
| Advanced  | 0.66      | 0.76   | 0.71     | 

**Overall Metrics:**

- **Accuracy:** 0.7225  


### Type Classification (Theory vs. Practice)

| Class     | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Theory    | 0.78      | 0.91   | 0.84     |
| Practice  | 0.84      | 0.64   | 0.73     | 

**Overall Metrics:**

- **Accuracy:** 0.7973 


# 4. Application
- The app fetched the books using **Google Books API** at real time
- Used coisine similarity to find related books based on the topic users input
- Use the fune-tuned BERT model predict the labels and output results, 3 books for each category

## Demo Link
[**Book Pairing System**](https://huggingface.co/spaces/kellly/Book-Pairing-System)

## Run Streamlit app locally

To run the code, run the following command:

```bash
streamlit run app.py
```

Click on the Local URL (http://localhost:8501) to open the Streamlit app in your browser.

# 5. Ethics statement
This project does not involve human subjects, personal data, or sensitive information. All book data used is publicly available through the Google Books API or open datasets. The system is designed solely for educational and informational purposes. Efforts were made to ensure fair and unbiased recommendations; however, model limitations may occasionally reflect bias present in the training data. We are committed to transparency and continuous improvement.