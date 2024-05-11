import argparse
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()

list_to_int = lambda s: [int(item) for item in s.split(',')]

parser.add_argument("--test_size", default=0.5, type=float, help="Velikost testovací množiny")
parser.add_argument('--one_hot', default=[], help="Index sloupců, kde má být použit one-hot encoder (číslujeme od 0)",
                    type=list_to_int)
parser.add_argument('--standard_scaler', default=[], help="Index sloupců, kde má být použit standard_scaler encoder (číslujeme od 0)",
                    type=list_to_int)
parser.add_argument('--polynomial_features', default=2, type=int, help="Maximální stupeň polynomiálních feature")
parser.add_argument("--seed", default=42, type=int, help="Náhodný seed")


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_diabetes()

    train_data, test_data, train_target, test_target = train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    ct = ColumnTransformer([
        ("one_hot", OneHotEncoder(), args.one_hot),
        ("standard_scaler", StandardScaler(), args.standard_scaler)],
        remainder='passthrough')

    poly = PolynomialFeatures(args.polynomial_features, include_bias=False)

    pipeline = Pipeline([
        ('column_transform', ct),
        ('polynomial_features', poly)])

    pipeline.fit(train_data)

    new_test_data = pipeline.transform(test_data)

    return test_data, new_test_data


def print_test_data(test_data):
    for line in range(2):
        print(" ".join("{:.4f}".format(test_data[line, column]) for column in range(test_data.shape[1])))


if __name__ == "__main__":
    args = parser.parse_args()
    prev_test_data, new_test_data = main(args)

    print("Original data:")
    print_test_data(prev_test_data)
    print("Transformed data:")
    print_test_data(new_test_data)