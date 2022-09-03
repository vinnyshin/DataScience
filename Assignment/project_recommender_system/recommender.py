import sys
import pandas as pd
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


def matrix_factorization(train_matrix, test_matrix, lmbd=0.02, k=1, std_dev=0.5, epochs=100, learning_rate=0.01):
    num_of_user = len(train_matrix.index)
    num_of_item = len(train_matrix.columns)

    u = std_dev * np.random.randn(num_of_user, k)  # user embedding
    v = std_dev * np.random.randn(num_of_item, k)  # item embedding

    bias_user = np.zeros((num_of_user, 1))
    bias_item = np.zeros((num_of_item, 1))

    matrix_mean = np.mean(train_matrix.values[~np.isnan(train_matrix.values)])

    train_not_nan_indices = train_matrix.stack(dropna=True).index.tolist()
    test_not_nan_indices = test_matrix.stack(dropna=True).index.tolist()

    train_set = [(i, j, train_matrix.loc[i, j]) for i, j in train_not_nan_indices]
    test_set = [(i, j, test_matrix.loc[i, j]) for i, j in test_not_nan_indices]

    train_user_item_matrix_index = train_matrix.index.tolist()
    train_user_item_matrix_columns = train_matrix.columns.tolist()

    train_avg_costs = []
    test_avg_costs = []

    for epoch in range(epochs):
        np.random.shuffle(train_set)

        train_avg_cost = 0
        test_avg_cost = 0

        for i, j, ground_truth in train_set:
            absolute_i = train_user_item_matrix_index.index(i)
            absolute_j = train_user_item_matrix_columns.index(j)

            logit = matrix_mean + bias_user[absolute_i] + bias_item[absolute_j] + u[absolute_i, :].dot(
                v[absolute_j, :].T)
            e = ground_truth - logit

            u[absolute_i, :] += learning_rate * (e * v[absolute_j, :] - lmbd * u[absolute_i, :])
            v[absolute_j, :] += learning_rate * (e * u[absolute_i, :] - lmbd * v[absolute_j, :])

            bias_user[absolute_i] += learning_rate * (e - lmbd * bias_user[absolute_i])
            bias_item[absolute_j] += learning_rate * (e - lmbd * bias_item[absolute_j])

            rms = sqrt(mean_squared_error([ground_truth], logit))
            train_avg_cost += rms / len(train_set)

        for i, j, ground_truth in test_set:
            if i not in train_user_item_matrix_index or j not in train_user_item_matrix_columns:
                continue

            absolute_i = train_user_item_matrix_index.index(i)
            absolute_j = train_user_item_matrix_columns.index(j)

            logit = matrix_mean + bias_user[absolute_i] + bias_item[absolute_j] + u[absolute_i, :].dot(
                v[absolute_j, :].T)

            test_rms = sqrt(mean_squared_error([ground_truth], logit))
            test_avg_cost += test_rms / len(test_set)

        train_avg_costs.append(train_avg_cost)
        test_avg_costs.append(test_avg_cost)

        print('Epoch: {} / {}\ntrain cost: {}\ntest cost: {}'.format(epoch + 1, epochs, train_avg_cost, test_avg_cost))

        if epoch > 0:
            if test_avg_costs[-2] < test_avg_cost:
                return matrix_mean, u, v, bias_user, bias_item

    return matrix_mean, u, v, bias_user, bias_item


def predict_and_write_to_file(mean, u, v, bias_user, bias_item, train_matrix, test_df, result_file_path):
    predicted_user_item_matrix = u.dot(v.T) + mean + bias_user + bias_item.T

    predicted_user_item_matrix[predicted_user_item_matrix < 0] = 0
    predicted_user_item_matrix[predicted_user_item_matrix > 5] = 5

    result_df = pd.DataFrame(predicted_user_item_matrix).apply(lambda x: np.round(x, 0))

    result_df = result_df.set_index(train_matrix.index)
    result_df.columns = train_matrix.columns

    for i in range(test_df.shape[0]):
        user_id, item_id = test_df.values[i][0], test_df.values[i][1]
        if user_id not in result_df.index:
            result_df.loc[user_id] = round(mean)
        if item_id not in result_df.columns:
            result_df[item_id] = round(mean)

    with open(result_file_path, 'w') as file:
        for a, b in result_df.stack().items():
            user = a[0]
            item = a[1]
            rating = b
            file.write(f"{user}\t{item}\t{rating}\n")


def run():
    start = time.time()

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    result_file_path = train_file_path + "_prediction.txt"
    columns = ["user_id", "item_id", "rating", "time_stamp"]

    train_df = pd.read_csv(train_file_path, names=columns, delimiter='\t')
    test_df = pd.read_csv(test_file_path, names=columns, delimiter='\t')

    # We are not going to use the last column "time_stamp".
    train_df.drop(columns=train_df.columns[-1], inplace=True)
    test_df.drop(columns=test_df.columns[-1], inplace=True)

    train_user_item_matrix = train_df.pivot(index='user_id', columns='item_id', values='rating')
    test_user_item_matrix = test_df.pivot(index='user_id', columns='item_id', values='rating')

    mean, u, v, bias_user, bias_item = matrix_factorization(train_user_item_matrix, test_user_item_matrix, k=1,
                                                            epochs=10, learning_rate=0.015, std_dev=0.5)

    predict_and_write_to_file(mean=mean, u=u, v=v, bias_user=bias_user, bias_item=bias_item,
                              train_matrix=train_user_item_matrix, test_df=test_df,
                              result_file_path=result_file_path)

    print("time :", time.time() - start)


if __name__ == "__main__":
    run()
