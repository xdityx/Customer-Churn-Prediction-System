from sklearn.metrics import classification_report , roc_auc_score

def evaluate_model(pipeline, X_test, y_test):
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:,1]


    print(classification_report(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, proba))
