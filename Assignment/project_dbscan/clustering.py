import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


def _distance(a, b):
    distances = pairwise_distances(a, b)
    return distances


def _get_directly_reachable_points(data, point, eps):
    '''
    :param data: pandas dataframe
    :param point: core point candidate
    :param eps: given eps(radius)
    :return: indices of all directly reachable point from candidate
    '''
    directly_reachable_idx = np.where(_distance(data, point) < eps)[0]
    #  eliminating self
    directly_reachable_idx = np.delete(directly_reachable_idx,
                                       np.argwhere(directly_reachable_idx == point.index.values))
    return directly_reachable_idx


def db_scan(data, eps, min_pts):
    current_cluster_number = 0

    label_vector = np.full(data.shape[0], np.NaN)
    label_vector = label_vector.reshape(-1, 1)

    while True:
        nan_mask = np.isnan(label_vector)
        nan_idx = np.where(nan_mask == True)[0]

        if len(nan_idx) == 0:
            break

        core_candidate_idx = np.random.choice(nan_idx, 1)
        core_candidate = data.iloc[core_candidate_idx]

        directly_reachable_points_idx = _get_directly_reachable_points(data, core_candidate, eps)

        if len(directly_reachable_points_idx) < min_pts:
            label_vector[core_candidate_idx] = -1.0  # means noise
            continue
        else:
            label_vector[core_candidate_idx] = current_cluster_number

        while len(directly_reachable_points_idx) > 0:
            next_candidate_idx = np.random.choice(directly_reachable_points_idx, 1)
            label_vector[next_candidate_idx] = current_cluster_number

            next_candidate = data.iloc[next_candidate_idx]

            next_candidate_directly_reachable_points_idx = _get_directly_reachable_points(data, next_candidate, eps)
            if len(next_candidate_directly_reachable_points_idx) >= min_pts:
                directly_reachable_points_idx = np.union1d(directly_reachable_points_idx,
                                                           next_candidate_directly_reachable_points_idx)

            condition1 = np.isnan(label_vector[directly_reachable_points_idx])
            condition1 = condition1.squeeze()
            condition2 = label_vector[directly_reachable_points_idx] == -1
            condition2 = condition2.squeeze()

            directly_reachable_points_idx = directly_reachable_points_idx[condition1 | condition2]

        current_cluster_number += 1

    data_labeled = pd.DataFrame(np.concatenate((data.values, label_vector), axis=1))
    return data_labeled


def write_result_to_disk(result_df, input_file_path, n_of_clusters):
    target_cluster_number = result_df[2].nunique() - n_of_clusters - 1

    clusters = result_df.groupby([2]).size()

    while target_cluster_number > 0:
        min_label_idx = clusters.argmin()
        min_label = clusters.index[min_label_idx]

        if min_label == -1:
            clusters.pop(min_label)
            min_label_idx = clusters.argmin()
            min_label = clusters.index[min_label_idx]

        result_df.loc[result_df[2] == min_label, 2] = -1
        target_cluster_number = result_df[2].nunique() - n_of_clusters - 1
        clusters = result_df.groupby([2]).size()

    basic_path = "./test-2/"
    for idx, num in enumerate(clusters.index):
        if num == -1:
            continue
        target = result_df[result_df[2] == num].index
        target = target + 1
        if idx == 0:  # means no outlier
            idx = idx + 1
        result_file_path = basic_path + "input" + input_file_path[-5] + f"_cluster_{idx - 1}.txt"
        with open(result_file_path, 'w') as file:
            for line in target:
                file.write(f"{line}\n")


def run():
    input_file_path = sys.argv[1]
    n_of_clusters = int(sys.argv[2])
    eps = int(sys.argv[3])
    min_pts = int(sys.argv[4])

    data = pd.read_csv(input_file_path, index_col=0, delimiter='\t')

    result_df = db_scan(data, eps, min_pts)

    # sns.scatterplot(data=result_df, x=0, y=1, hue=2, palette='Set2')
    # plt.show()

    write_result_to_disk(result_df, input_file_path, n_of_clusters)


if __name__ == "__main__":
    run()

