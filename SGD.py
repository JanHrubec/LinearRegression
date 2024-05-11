import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import math

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--l2", default=0.0, type=float, help="Síla L2 regularizace")
parser.add_argument("--epochs", default=50, type=int, help="Počet epoch na trénování Minibatch SGD")
parser.add_argument("--batch_size", default=10, type=int, help="Velikost batche")
parser.add_argument("--data_size", default=100, type=int, help="Velikost datasetu")
parser.add_argument("--test_size", default=0.5, type=float, help="Velikost testovací množiny")
parser.add_argument("--seed", default=42, type=int, help="Náhodný seed")
parser.add_argument("--plot", action='store_true', help="Vykreslit predikce")

def main(args: argparse.Namespace):
    generator = np.random.RandomState(args.seed)

    data, target = make_regression(n_samples=args.data_size, random_state=args.seed)

    data = np.concatenate([data, np.ones([args.data_size, 1])], axis=1)

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    weights = generator.uniform(size=train_data.shape[1])

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        perm = generator.permutation(train_data.shape[0])
        batch_num = math.ceil(train_data.shape[0] / args.batch_size)
        for batch in range(batch_num):
            batch_predictions = []
            batch_data = []
            batch_target = []
            for i in range(args.batch_size):
                p = perm[batch * args.batch_size + i]
                data = train_data[p]
                target = train_target[p]
                prediction = data.dot(weights)

                batch_predictions.append(prediction)
                batch_data.append(data)
                batch_target.append(target)
                
            gradient = np.zeros(weights.shape)
            for weight_idx in range(weights.shape[0]):
                for i in range(args.batch_size):
                    gradient[weight_idx] += 1/args.batch_size * 2 * (batch_predictions[i] - batch_target[i]) * batch_data[i][weight_idx]

            weights -= args.learning_rate * gradient

            if args.l2 != 0:
                weights -= args.learning_rate * args.l2 * 2 * weights

        train_predictions = []
        for data in train_data:
            prediction = data.dot(weights)
            train_predictions.append(prediction)
        test_predictions = []
        for data in test_data:
            prediction = data.dot(weights)
            test_predictions.append(prediction)
        train_rmse, test_rmse = mean_squared_error(train_target, train_predictions, squared = False), mean_squared_error(test_target, test_predictions, squared = False)

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
        print(f"Epoch {epoch+1}: train = {train_rmse:.8f}, test = {test_rmse:.8f}")

    model = Ridge(alpha=args.l2, fit_intercept=False)
    model.fit(train_data, train_target)
    model_predictions = model.predict(test_data)
    test_sklearn_rmse = mean_squared_error(test_target, model_predictions, squared = False)
    print(f"Sklearn RMSE = {test_sklearn_rmse:.8f}")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Trénovací chyba (RMSE)")
        plt.plot(test_rmses, label="Testovací chyba (RMSE)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)