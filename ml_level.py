import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def train_with_grid_search():
    # Step 1: Load and prepare data
    df = pd.read_csv("data/labeled_books.csv")
    df = df[df["level"].isin(["Beginner", "Advanced"])].dropna(subset=["description"]).reset_index(drop=True)
    label_map = {"Beginner": 0, "Advanced": 1}
    df["label"] = df["level"].map(label_map)

    # Step 2: TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df["description"])
    y = df["label"]

    # Step 3: Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 4: Define parameter grid and CV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
    }

    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='f1', cv=5, n_jobs=-1)

    # Step 5: Run grid search on training set
    grid.fit(X_train, y_train)

    print("=== Best Hyperparameters from Grid Search ===")
    print(grid.best_params_)
    print("Best CV F1 Score:", grid.best_score_)
    print("")

    # Step 6: Final evaluation on test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("=== Final Evaluation on Test Set ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=["Beginner", "Advanced"]))

if __name__ == "__main__":
    train_with_grid_search()
