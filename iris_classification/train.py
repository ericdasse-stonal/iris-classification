from joblib import dump
import pandas as pd
from sklearn.linear_model import LogisticRegression

from iris_classification.settings import MODEL_DIR, PREPARED_DATA_DIR


def train(prepared_data_path, model_dir, serialize=True):
    iris_train_data = pd.read_csv(prepared_data_path)
    X_train = iris_train_data.drop("species", axis=1)
    y_train = iris_train_data["species"]

    # Train with Logistic Regression
    log_reg = LogisticRegression()
    iris_model = log_reg.fit(X_train, y_train)

    if serialize:
        dump(iris_model, model_dir)


if __name__ == "__main__":
    train(
        prepared_data_path=PREPARED_DATA_DIR / "train.csv",
        model_dir=MODEL_DIR,
    )
