from data_loader import load_data
from preprocessing import clean_data
from feature_engineering import create_features
from train import train_rf
from evaluate import evaluate_model
from threshold_tuning import tune_threshold
from config import target_col, test_size, random_state
from sklearn.model_selection import train_test_split
from preprocessing_pipeline import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_PATH = "data/raw/telecom_churn.csv"
MODEL_PATH = "models/churn_model.pkl"

def main():
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Clean
    df = clean_data(df)
    # REMOVE high-cardinality columns
    df.drop(columns=["city", "pincode"], errors="ignore", inplace=True)

    # 3. Feature engineering (business logic)
    df = create_features(df,mode = 'train')

    # 4. Split
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 5. Build preprocessing (ENCODING + SCALING)
    preprocessor, _, _ = build_preprocessor(X)

    # 6. Full pipeline (THIS IS WHERE PIPELINE LIVES)
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state
        ))
    ])

    # 7. Train
    pipeline.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Pipeline trained successfully")

        # Threshold tuning
    threshold_results = tune_threshold(y_test, y_proba)

    print("\nThreshold Tuning Results:")
    for r in threshold_results:
        print(r)


    # 9. Save pipeline (VERY IMPORTANT)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

if __name__ == "__main__":
    main()
