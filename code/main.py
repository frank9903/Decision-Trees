import numpy as np
import os
import csv
from collections import Counter
from anytree.exporter import DotExporter
from anytree import Node
from scipy import stats
import matplotlib.pyplot as plt

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
    global labels
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
    global node_num
    node_num += 1
    qualities = np.array(examples)[:,-1]
    if not np.any(qualities != qualities[0]):
        n = Node(str(node_num)+": "+str(qualities[0]), v=qualities[0])
        return n
    else:
        f, v, h = choose_feature(examples, features)
        ex1, ex2 = split_examples(examples, f, v)
        n = Node(str(node_num)+": "+dictionary[f]+"<="+str(v), parent=tree, v=[f,v])
        n.children = [learn_dt(n, ex1, features), learn_dt(n, ex2, features)]
        return n

def learn_dt_max_height(tree, examples, max_height, features, cur_height=1):
    global node_num
    node_num+=1
    qualities = np.array(examples)[:,-1]
    if (cur_height >= max_height):
        q = stats.mode(qualities)[0][0]
        n = Node(str(node_num)+": "+str(q), v=q)
        return n
    if not np.any(qualities != qualities[0]):
        n = Node(str(node_num)+": "+str(qualities[0]), v=qualities[0])
        return n
    else:
        f, v, h = choose_feature(examples, features)
        ex1, ex2 = split_examples(examples, f, v)
        n = Node(str(node_num)+": "+dictionary[f]+"<="+str(v), parent=tree, v=[f,v])
        n.children = [learn_dt_max_height(n, ex1, max_height, features, cur_height+1), learn_dt_max_height(n, ex2, max_height, features, cur_height+1)]
        return n

def cross_validation(data, heights, kfolds, features):
    targets = np.array(data)[:,-1]
    splited_data = [data[:319], data[319:638], data[638:957], data[957:1276], data[1276:]]
    splited_tar = [targets[:319], targets[319:638], targets[638:957], targets[957:1276], targets[1276:]]
    best_max_height = 0
    best_train_acc = 0
    best_vali_acc = 0
    train_accs = []
    vali_accs = []
    start = True
    for max_height in heights:
        temp_train_acc = 0
        temp_vali_acc = 0
        for i in range(kfolds):
            validation_x = splited_data[i]
            validation_y = splited_tar[i]
            train_x = np.concatenate(splited_data[:i] + splited_data[i+1:])
            train_y = np.concatenate(splited_tar[:i] + splited_tar[i+1:])
            tree = learn_dt_max_height(None, list(train_x), max_height, features)
            temp_train_acc += get_prediction_accuracy(tree, train_x)
            temp_vali_acc += get_prediction_accuracy(tree, validation_x)
        avg_train_acc = float(temp_train_acc) / kfolds
        avg_vali_acc = float(temp_vali_acc) / kfolds
        train_accs.append(avg_train_acc)
        vali_accs.append(avg_vali_acc)
        if (start):
            best_vali_acc = avg_vali_acc
            best_max_height = max_height
            start = False
        if (avg_vali_acc > best_vali_acc):
            best_vali_acc = avg_vali_acc
            best_max_height = max_height
    return best_max_height, train_accs, vali_accs, best_train_acc, best_vali_acc

def predict(tree, example):
    cur_node = tree
    while (not cur_node.is_leaf):
        val = cur_node.v
        if (example[val[0]] <= val[1]):
            cur_node = cur_node.children[0]
        else:
            cur_node = cur_node.children[1]
    return cur_node.v

def get_prediction_accuracy(tree, data):
    total = len(data)
    correct = 0
    for row in data :
        if predict(tree, row[:-1]) == row[-1]:
            correct += 1
    return correct / float(total)
    
def plot_2b(train_accs, test_accs, heights):
    plt.figure()
    plt.plot(heights,train_accs, color='blue', label='Training accuracies')
    plt.plot(heights,test_accs, color='red', label='Validation accuracies')
    plt.legend(['Training accuracies', 'Validation accuracies'], loc='upper left')
    plt.ylabel('accuracies')
    plt.xlabel('max heights')
    plt.savefig('cv-max-depth.png')

def learn_dt_post_pruning(tree, examples, min_info_gain, features):
    global node_num
    node_num += 1
    qualities = np.array(examples)[:,-1]
    if not np.any(qualities != qualities[0]):
        n = Node(str(node_num)+": "+str(qualities[0]), v=qualities[0])
        return n
    else:
        f, v, h = choose_feature(examples, features)
        ex1, ex2 = split_examples(examples, f, v)
        n = Node(str(node_num)+": "+dictionary[f]+"<="+str(v), parent=tree, v=[f,v])
        child1 = learn_dt_post_pruning(n, ex1, min_info_gain, features)
        child2 = learn_dt_post_pruning(n, ex2, min_info_gain, features)
        if child1.is_leaf and child2.is_leaf and h < min_info_gain:
            q = stats.mode(qualities)[0][0]
            n = Node(str(node_num)+": "+str(q), v=q)
            return n
        n.children = [child1, child2]
        return n

def cross_validation_2c(data, min_info_gains, kfolds, features):
    global node_num
    targets = np.array(data)[:,-1]
    splited_data = [data[:319], data[319:638], data[638:957], data[957:1276], data[1276:]]
    splited_tar = [targets[:319], targets[319:638], targets[638:957], targets[957:1276], targets[1276:]]
    best_min_info_gain = 0
    best_train_acc = 0
    best_vali_acc = 0
    train_accs = []
    vali_accs = []
    start = True
    for min_info_gain in min_info_gains:
        temp_train_acc = 0
        temp_vali_acc = 0
        for i in range(kfolds):
            validation_x = splited_data[i]
            validation_y = splited_tar[i]
            train_x = np.concatenate(splited_data[:i] + splited_data[i+1:])
            train_y = np.concatenate(splited_tar[:i] + splited_tar[i+1:])
            node_num = 0
            tree = learn_dt_post_pruning(None, list(train_x), min_info_gain, features)
            temp_train_acc += get_prediction_accuracy(tree, train_x)
            temp_vali_acc += get_prediction_accuracy(tree, validation_x)
        avg_train_acc = float(temp_train_acc) / kfolds
        avg_vali_acc = float(temp_vali_acc) / kfolds
        train_accs.append(avg_train_acc)
        vali_accs.append(avg_vali_acc)
        if (start):
            best_vali_acc = avg_vali_acc
            best_min_info_gain = min_info_gain
            start = False
        if (avg_vali_acc > best_vali_acc):
            best_vali_acc = avg_vali_acc
            best_min_info_gain = min_info_gain
    return best_min_info_gain, train_accs, vali_accs, best_train_acc, best_vali_acc

def plot_2c(train_accs, test_accs, min_info_gains):
    plt.figure()
    plt.plot(min_info_gains, train_accs, color='blue', label='Training accuracies')
    plt.plot(min_info_gains, test_accs, color='red', label='Validation accuracies')
    plt.legend(['Training accuracies', 'Validation accuracies'], loc='upper left')
    plt.ylabel('accuracies')
    plt.xlabel('min_info_gains')
    plt.savefig('cv-min-info-gain.png')

if __name__ == '__main__':
    node_num = 0
    data = read_data("winequality_red_comma.csv")
    tree_full = learn_dt(None, data, range(len(data[0])-1))
    acc = get_prediction_accuracy(tree_full, data)
    h = tree_full.height + 1
    DotExporter(tree_full).to_picture('tree-full.png')
    print("Maximum depth for 2a is: " + str(h))
    print("Prediction accuracy for 2a is: "+str(acc))

    node_num = 0
    data = read_data("winequality_red_comma.csv")
    heights = range(1,18)
    best_max_height, train_accs, vali_accs, best_train_acc, best_vali_acc = cross_validation(data, heights, 5, range(len(data[0])-1))
    print("Best value of the maximum depth of the tree for 2b is: " + str(best_max_height))
    data = read_data("winequality_red_comma.csv")
    node_num = 0
    tree_max_depth = learn_dt_max_height(None, data, best_max_height, range(len(data[0])-1))
    acc = get_prediction_accuracy(tree_max_depth, data)
    DotExporter(tree_max_depth).to_picture('tree-max-depth.png')
    print("Prediction accuracy for 2b is: "+str(acc))
    plot_2b(train_accs, vali_accs, heights)

    node_num = 0
    data = read_data("winequality_red_comma.csv")
    min_info_gains = np.arange(0.0, 1.5, 0.05)
    best_min_info_gain, train_accs, vali_accs, best_train_acc, best_vali_acc = cross_validation_2c(data, min_info_gains, 5, range(len(data[0])-1))
    print("Best value of the minimum information gain for 2c is: " + str(best_min_info_gain))
    node_num = 0
    tree_min_info_gain = learn_dt_post_pruning(None, data, best_min_info_gain, range(len(data[0])-1))
    acc = get_prediction_accuracy(tree_min_info_gain, data)
    DotExporter(tree_min_info_gain).to_picture('tree-min-info-gain.png')
    print("Prediction accuracy for 2c is: "+str(acc))
    plot_2c(train_accs, vali_accs, min_info_gains)

