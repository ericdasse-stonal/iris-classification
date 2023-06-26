import pandas as pd

from iris_classification.settings import PREPARED_DATA_PATH, RAW_DATA_PATH


def prepare(raw_data_path, prepared_data_path):
    # Load raw dataset
    iris = pd.read_csv(
        raw_data_path,
        header=None,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    )

    # Simplify species names by removing the prefix "Iris-"
    iris["species"] = iris["species"].apply(lambda s: s.replace("Iris-", ""))

    iris.to_csv(prepared_data_path)


if __name__ == "__main__":
    prepare(raw_data_path=RAW_DATA_PATH, prepared_data_path=PREPARED_DATA_PATH)
