import sys
import csv
from math import ceil
from itertools import combinations
from functools import reduce

TAB_DIALECT = 'tab_dialect'
csv.register_dialect(TAB_DIALECT, delimiter='\t')


def mine_association_rule(transaction_db, total_frequent_patterns):
    association_rules = []

    for frequent_pattern in total_frequent_patterns:
        for given_event_item_count in range(1, len(frequent_pattern[0])):
            frozenset_frequent_pattern = frequent_pattern[0]
            for given_event in combinations(frozenset_frequent_pattern, given_event_item_count):
                given_event_in_transaction_count = 0
                confidence_count = 0
                given_event_set = frozenset(given_event)
                complement_of_given_event = frozenset_frequent_pattern.difference(given_event_set)

                for transaction in transaction_db:
                    flag = True
                    for element in given_event:
                        if element not in transaction:
                            flag = False
                            break
                    if flag:
                        given_event_in_transaction_count += 1
                        confidence_flag = True
                        for element in complement_of_given_event:
                            if element not in transaction:
                                confidence_flag = False
                                break
                        if confidence_flag:
                            confidence_count += 1

                confidence = (confidence_count / given_event_in_transaction_count) * 100
                support = (frequent_pattern[1] / transaction_db.__len__()) * 100
                association_rules.append((given_event, tuple(complement_of_given_event), support, confidence))

    return association_rules


def apriori(transaction_db, frequent_patterns_1, support_to_count):
    candidate = {}
    current_frequent_patterns = frequent_patterns_1
    total_frequent_patterns = []

    trial = 1

    while True:
        for combination in combinations(current_frequent_patterns.keys(), 2):
            combination_set = reduce(frozenset.union, combination)

            if combination_set.__len__() != trial + 1:
                continue

            is_candidate = True

            if trial != 1:
                for sub_combination in combinations(combination_set, trial):
                    sub_combination_set = frozenset(sub_combination)

                    flag = False

                    for frequent_pattern_set in current_frequent_patterns:
                        if sub_combination_set == frequent_pattern_set:
                            flag = True
                            break

                    if not flag:
                        is_candidate = False

            if is_candidate:
                candidate[combination_set] = 0

        for item_set in candidate:  # calculate support
            for transaction in transaction_db:
                flag = True
                for item in item_set:
                    if item not in transaction:
                        flag = False
                        break
                if flag:
                    candidate[item_set] += 1

        current_frequent_patterns = dict(filter(lambda x: x[1] >= support_to_count, candidate.items()))
        total_frequent_patterns.extend(list(current_frequent_patterns.items()))
        candidate.clear()

        trial += 1

        if len(current_frequent_patterns) == 0:
            break

    return total_frequent_patterns


def create_txt(csv_writer, association_rules):
    for association in association_rules:
        p = f'{{{",".join(map(lambda x: str(x), association[0]))}}}'
        q = f'{{{",".join(map(lambda x: str(x), association[1]))}}}'

        support = f'{association[2]:.2f}'
        confidence = f'{association[3]:.2f}'

        csv_writer.writerow([p, q, support, confidence])


def run():
    support = int(sys.argv[1]) * 0.01  # measurement is %
    input_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    input_file = open(input_file_name, 'r')
    output_file = open(output_file_name, 'w', newline="")

    csv_reader = csv.reader(input_file, TAB_DIALECT)
    csv_writer = csv.writer(output_file, TAB_DIALECT, delimiter='\t')

    transaction_db = []
    candidate_patterns_1 = {}  # candidate_k means Candidate item set of size k

    for transaction in csv_reader:
        transaction = frozenset(map(lambda x: int(x), transaction))

        for item in transaction:
            frozenset_item = frozenset({item})
            if candidate_patterns_1.get(frozenset_item) is None:
                candidate_patterns_1[frozenset_item] = 1
            else:
                candidate_patterns_1[frozenset_item] += 1
        transaction_db.append(transaction)

    support_to_count = ceil(transaction_db.__len__() * support)

    frequent_patterns_1 = dict(filter(lambda x: x[1] >= support_to_count, candidate_patterns_1.items()))

    total_frequent_patterns = apriori(transaction_db, frequent_patterns_1, support_to_count)
    association_rules = mine_association_rule(transaction_db, total_frequent_patterns)
    create_txt(csv_writer, association_rules)

    input_file.close()
    output_file.close()


if __name__ == "__main__":
    run()

