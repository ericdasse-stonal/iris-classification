from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from iris_classification.settings import IRIS_COLUMNS, PREPARED_DATA_DIR, RAW_DATA_DIR


def save_as_csv(X, y, destination):
    iris_features = pd.DataFrame(X)
    labels = pd.Series(y)

    iris_data: pd.DataFrame = pd.concat([iris_features, labels], axis=1)
    iris_data.columns = IRIS_COLUMNS[2:]
    iris_data.to_csv(destination, index=False)


def prepare(raw_data_path: Path, prepared_data_dir: Path):
    # Load raw dataset
    iris = pd.read_csv(
        raw_data_path,
        header=None,
        names=IRIS_COLUMNS,
    )

    # Simplify species names by removing the prefix "Iris-"
    iris["species"] = iris["species"].apply(lambda s: s.replace("Iris-", ""))

    # Split the data into training and test
    X = iris.drop("species", axis=1)
    X = X.to_numpy()[:, (2, 3)]
    y = iris["species"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Save the training and test data to CSV files
    save_as_csv(X_train, y_train, prepared_data_dir / "train.csv")
    save_as_csv(X_test, y_test, prepared_data_dir / "test.csv")


if __name__ == "__main__":
    prepare(raw_data_path=RAW_DATA_DIR, prepared_data_dir=PREPARED_DATA_DIR)
