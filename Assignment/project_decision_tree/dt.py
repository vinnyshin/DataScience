import sys
import pandas as pd
import numpy as np


class DecisionTreeNode:
    def __init__(self, test_attribute_name=None, label=None):
        self.children = {}
        self.test_attribute_name = test_attribute_name
        self.label = label  #type: str


def calculate_entropy(df, column_idx):
    probabilities = df.groupby(df.columns[column_idx]).size().div(len(df))
    entropy = probabilities.apply(lambda x: -x * np.log2(x)).sum()
    return entropy


def select_test_attribute(df, df_entropy):
    test_attribute = (None, -1)

    for column_idx in range(0, df.columns.size - 1):
        split_info = 0
        entropy = 0
        column_name = df.columns[column_idx]
        for key, local_df in df.groupby(column_name):
            probability = local_df.__len__() / df.__len__()
            split_info += -probability * np.log2(probability)
            entropy += probability * calculate_entropy(local_df, -1)
        delta_entropy = df_entropy - entropy
        gain_ratio = delta_entropy / split_info
        if test_attribute[1] < gain_ratio:
            test_attribute = (column_name, gain_ratio)

    return test_attribute[0]


def decision_tree(df, root):
    global column_name_value_pair

    if df.size == 0:
        root.label = None
        return root

    df_entropy = calculate_entropy(df, column_idx=-1)

    if df_entropy == 0:  # All tuples have same class label.
        root.label = df.iloc[-1, -1]
        return root

    root.test_attribute_name = select_test_attribute(df, df_entropy)
    majority = df.iloc[:, -1].mode().iat[0]

    if root.test_attribute_name is None:  # No more attributes for test attribute.
        root.label = majority
        return root

    for key, local_df in df.groupby(root.test_attribute_name):
        root.children[key] = decision_tree(local_df.drop(root.test_attribute_name, axis=1), DecisionTreeNode())

    for value in column_name_value_pair[root.test_attribute_name]:
        if root.children.get(value) is None:
            root.children[value] = DecisionTreeNode(test_attribute_name=root.test_attribute_name, label=majority)

            # for key, values in column_name_value_pair.items():
    #     if key == root.test_attribute_name:
    #         for value in values:
    #             if root.children.get(value) is None:
    #                 root.children[value] = DecisionTreeNode(test_attribute_name=root.test_attribute_name, label=majority)

    return root


def traverse_tree(root, row):
    while root.label is None:
        root = root.children[row[root.test_attribute_name]]
    return root.label


def predict_class_label(test_df, root):
    label_vector = test_df.apply(lambda x: traverse_tree(root, x), axis=1)
    return label_vector


column_name_value_pair = {}


def run():
    global column_name_value_pair
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    result_file_path = sys.argv[3]

    train_df = pd.read_csv(train_file_path, index_col=None, delimiter='\t')

    for column_name, column_val in train_df.iteritems():
        column_name_value_pair[column_name] = column_val.unique().tolist()

    root = DecisionTreeNode()
    root = decision_tree(train_df, root)

    test_df = pd.read_csv(test_file_path, index_col=None, delimiter='\t')

    result_vector = predict_class_label(test_df, root)
    class_label_attribute_name = train_df.columns[-1]
    result_vector.name = class_label_attribute_name

    result_df = pd.concat([test_df, result_vector], axis=1)
    result_df.to_csv(result_file_path, sep='\t', index=False)


if __name__ == "__main__":
    run()
