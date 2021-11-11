import numpy as np
import csv
from collections import Counter
import sys
import unittest
from anytree import Node

labels = []
dictionary = {
    0:"fixed acidity",
    1:"volatile acidity",
    2:"citric acid",
    3:"residual sugar",
    4:"chlorides",
    5:"free sulfur dioxide",
    6:"total sulfur dioxide",
    7:"density",
    8:"pH",
    9:"sulphates",
    10:"alcohol"
}

def read_data(file_path):
    data = []
    start = False
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if start:
                temp = float(row[-1])
                if not temp in labels:
                    labels.append(temp)
            start = True
            data.append(row)
    labels.sort()
    return [[float(y) for y in x] for x in data[1:]]

def have_diff(c1, c2):
    l1 = []
    l2 = []
    for i in range(len(c1)):
        if c1[i] != 0:
            l1.append(i)
        if c2[i] != 0:
            l2.append(i)
    for a in l1:
        for b in l2:
            if a != b:
                return True
    return False

def get_splits(examples, feature):
    global labels
    examples.sort(key=lambda x : x[feature])
    result = {}
    cumulate = [0 for x in labels]
    prev_feature_value = examples[0][feature]
    prev_count = []
    cur_feature_value = prev_feature_value
    cur = []
    cur_count = [0 for x in labels]
    for row in examples:
        if (row[feature] != cur_feature_value):
            for i in range(len(labels)):
                cur_count[i] = cur.count(labels[i])
            if prev_count:
                cumulate = np.array(cumulate) + np.array(prev_count)
                if (have_diff(prev_count, cur_count)):
                    result[(prev_feature_value + cur_feature_value) / 2] = cumulate
            prev_count = cur_count
            cur_count = [0 for x in labels]
            cur = []
            prev_feature_value = cur_feature_value
            cur_feature_value = row[feature]
        cur.append(row[-1])
    if len(prev_count) > 0:
        for i in range(len(labels)):
            cur_count[i] = cur.count(labels[i])
        cumulate = np.array(cumulate) + np.array(prev_count)
        if (have_diff(prev_count, cur_count)):
            result[(prev_feature_value + cur_feature_value) / 2] = cumulate
    print(result)
    return result

def split_examples(examples, feature, split):
    ex1 = []
    ex2 = []
    for row in examples:
        if row[feature] <= split:
            ex1.append(row)
        else:
            ex2.append(row)
    return ex1, ex2

def entropy(distribution):
    distribution = [value for value in distribution if value != 0]
    dist = np.array(distribution)
    return -1 * np.sum(dist * np.log2(dist))

def calc_entropy(examples):
    cnt = Counter(np.array(examples)[:,-1])
    cnt = np.array(list(cnt.values()))
    total = np.sum(cnt)
    distribution = cnt / total
    return entropy(distribution)

def choose_split(examples, feature, debug=False):
    global labels
    cnt = Counter(np.array(examples)[:,-1])
    cnt = np.array([cnt[i] for i in labels])
    total = len(examples)
    h_before = entropy(cnt / total)
    splits = get_splits(examples, feature)
    cur_max = "NONE"
    best_value = 0
    for key, value in splits.items():
        value = np.array(value)
        value2 = cnt - value
        total1 = np.sum(value)
        distribution1 = value / total1
        total2 = np.sum(value2)
        distribution2 = value2 / total2
        h_after = ((total1 / total) * entropy(distribution1)) + ((total2 / total) * entropy(distribution2))
        information_gain = h_before - h_after
        if debug:
            print(dictionary[feature]+": "+str(key)+", info_gain: "+str(information_gain))
        if cur_max == "NONE":
            cur_max = information_gain
            best_value = key
        else:
            if (information_gain > cur_max):
                cur_max = information_gain
                best_value = key
    return cur_max, best_value

def choose_feature(example, features, debug=False):
    cur_max = "NONE"
    best_value = 0
    best_feature = 0
    for feature in features:
        h, f_value = choose_split(example, feature, debug)
        if h == "NONE":
            continue
        if cur_max == "NONE":
            cur_max = h
            best_value = f_value
            best_feature = feature
        else:
            if (h > cur_max):
                cur_max = h
                best_value = f_value
                best_feature = feature
    return best_feature, best_value, cur_max

node_num = -1

def learn_dt(tree, examples, features):
    print("--------------------------------")
    global node_num
    node_num += 1
    print(examples)
    qualities = np.array(examples)[:,-1]
    if not np.any(qualities != qualities[0]):
        n = Node(str(node_num)+": "+str(qualities[0]), v=qualities[0])
        return n
    else:
        f, v, h = choose_feature(examples, features)
        ex1, ex2 = split_examples(examples, f, v)
        n = Node(str(node_num)+": "+dictionary[f]+"<="+str(v)+", inf_gain: "+str(h), parent=tree, v=[f,v])
        n.children = [learn_dt(n, ex1, features), learn_dt(n, ex2, features)]
        return n

def predict(tree, example):
    cur_node = tree
    while (not cur_node.is_leaf):
        val = cur_node.v
        if (example[val[0]] <= val[1]):
            cur_node = cur_node.children[0]
        else:
            cur_node = cur_node.children[1]
    return cur_node.v

def round_keys(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[round(k, 2)] = v
    return new_dict

tol = 1e-10

class TestMethods(unittest.TestCase):
    
    def test_read_data(self):
        data = read_data("iris_reduced1.csv")
        r1 = [[5.1, 3.5, 1.4, 0.2, 0.0], [4.9, 3.0, 1.4, 0.2, 0.0], [4.7, 3.2, 1.3, 0.2, 0.0], [4.6, 3.1, 1.5, 0.2, 0.0], [7.0, 3.2, 4.7, 1.4, 1.0], [6.4, 3.2, 4.5, 1.5, 1.0], [6.9, 3.1, 4.9, 1.5, 1.0], [5.5, 2.3, 4.0, 1.3, 1.0]]
        self.assertTrue(data == r1)
        print("test_read_data passed")

    def test_get_splits(self):
        # return value format for get_splits, a dictionary
        # keys: split point value; values: a array of the number of labels dimension, first number represents the number of examples less than the split point and have label 0, second number represents the number of examples less than the split point and have label 1
        data = read_data("iris_reduced1.csv")
        s1 =  get_splits(data, 0)
        self.assertTrue(s1, {5.3: [4, 0]})
        s2 = get_splits(data, 1)
        self.assertTrue(sorted(round_keys(s2)), sorted({2.65: [0, 1], 3.05: [1, 1], 3.15: [2, 2], 3.35: [3, 4]}))
        print("test_get_splits passed")

    def test_calc_entropy(self):
        # note: the first three tests can be removed if you don't have an entropy function written in this way
        e1 = entropy([0.2,0.2,0.6])
        self.assertTrue(1.37095059445-tol < e1 < 1.37095059445+tol)
        e2 = entropy([0.1,0,0.9])
        self.assertTrue(0.46899559358-tol < e2 < 0.46899559358+tol)
        e3 = entropy([1,0,0])
        self.assertTrue(0-tol < e3 < 0+tol)
        examples = read_data("iris_reduced1.csv")
        e4 = calc_entropy(examples)
        self.assertTrue(1-tol < e4 < 1+tol)
        print("test_calc_entropy passed")

    def test_split_examples(self):
        data = read_data("iris_reduced1.csv")
        ex1, ex2 = split_examples(data, 0, 5.3)
        self.assertTrue(sorted(ex1) == sorted([[5.1, 3.5, 1.4, 0.2, 0.0], [4.9, 3.0, 1.4, 0.2, 0.0], [4.7, 3.2, 1.3, 0.2, 0.0], [4.6, 3.1, 1.5, 0.2, 0.0]]) \
                        and sorted(ex2) == sorted([[7.0, 3.2, 4.7, 1.4, 1.0], [6.4, 3.2, 4.5, 1.5, 1.0], [6.9, 3.1, 4.9, 1.5, 1.0], [5.5, 2.3, 4.0, 1.3, 1.0]]))
        print("test_split_examples passed")

    def test_choose_feature(self):
        data = read_data("iris_reduced1.csv")
        f1, sp1, h = choose_feature(data, [0])
        self.assertTrue(f1 == 0 and sp1 == 5.3)
        f2, sp2, h = choose_feature(data, [1])
        print(f2, sp2)
        self.assertTrue(f2 == 1 and (sp2 == 3.35 or sp2 == 2.65))
        f3, sp3, h = choose_feature(data, [1, 0])
        self.assertTrue(f3 == 0 and sp3 == 5.3)
        print("test_choose_feature passed")

    def test_predict(self):
        data1 = read_data("iris_reduced1.csv")
        full_tree = learn_dt(None, data1, [0,1,2,3]) # note: add "features" parameter if necessary
        d1 = predict(full_tree, [1,0,0,0,0])
        self.assertTrue(d1 == 0)
        data2 = read_data("iris_reduced2.csv")
        full_tree1 = learn_dt(None, data2, [0,1,2,3])
        d2 = predict(full_tree1, [1,0,0,0,0])
        self.assertTrue(d2 == 0)
        d3 = predict(full_tree1, [5.4, 2.6, 0, 1])
        print(d3)
        self.assertTrue(d3 == 1)
        print("test_predict passed")

if __name__ == '__main__':
    unittest.main()
