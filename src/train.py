from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



def split_data(X, y , test_size,random_state):
    return train_test_split(
        X, y ,
        test_size = test_size,
        random_state = random_state,
        stratify = y
    )


def train_baseline(X_train,y_train):
    model = LogisticRegression(max_iter = 1000, class_weights = "balanced")
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators = 500 ,
        class_weight = "balanced",
        random_state = 42)