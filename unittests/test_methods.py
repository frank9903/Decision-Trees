import sys
import unittest

################## should be changed depending on your implementation ###################
base_path = '/Users/wanxinli/Desktop/Fall 2020/TA/A2/code/' 
sys.path.insert(1,base_path)
import config
config.label_dim = 2 # number of classes in iris_reduced.csv
from dt import entropy, read_data, get_splits, choose_feature,split_examples, learn_dt, init_tree, predict, calc_entropy
index_dict = {
    'sepal_len_index': 0,
    'sepal_width_index': 1,
    'petal_len_index': 2,
    'petal_width_index': 3,
    'class_index': 4
}
##########################################################################################

# unit tests 
tol = 1e-10

def round_keys(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[round(k, 2)] = v
    return new_dict

class TestMethods(unittest.TestCase):
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
    
    def test_read_data(self):
        data = read_data("iris_reduced1.csv")
        r1 = [[5.1, 3.5, 1.4, 0.2, 0.0], [4.9, 3.0, 1.4, 0.2, 0.0], [4.7, 3.2, 1.3, 0.2, 0.0], [4.6, 3.1, 1.5, 0.2, 0.0], [7.0, 3.2, 4.7, 1.4, 1.0], [6.4, 3.2, 4.5, 1.5, 1.0], [6.9, 3.1, 4.9, 1.5, 1.0], [5.5, 2.3, 4.0, 1.3, 1.0]]
        self.assertTrue(data == r1)
        print("test_read_data passed")

    def test_get_splits(self):
        # return value format for get_splits, a dictionary
        # keys: split point value; values: a array of the number of labels dimension, first number represents the number of examples less than the split point and have label 0, second number represents the number of examples less than the split point and have label 1
        data = read_data("iris_reduced1.csv")
        s1 =  get_splits(data, 'sepal_len')
        self.assertTrue(s1, {5.3: [4, 0]})
        s2 = get_splits(data, 'sepal_width')
        self.assertTrue(sorted(round_keys(s2)), sorted({2.65: [0, 1], 3.05: [1, 1], 3.15: [2, 2], 3.35: [3, 4]})) 
        print("test_get_splits passed")

    def test_choose_feature(self):
        data = read_data("iris_reduced1.csv")
        f1, sp1 = choose_feature(['sepal_len'], data)
        self.assertTrue(f1 == 'sepal_len' and sp1 == 5.3)
        f2, sp2 = choose_feature(['sepal_width'], data)
        self.assertTrue(f2 == 'sepal_width' and (sp2 == 3.35 or sp2 == 2.65))
        f3, sp3 = choose_feature(['sepal_width', 'sepal_len'], data)
        self.assertTrue(f3 == 'sepal_len' and sp3 == 5.3)
        print("test_choose_feature passed")
    
    def test_split_examples(self):
        data = read_data("iris_reduced1.csv")
        ex1, ex2 = split_examples(data, 'sepal_len', 5.3)
        self.assertTrue(sorted(ex1) == sorted([[5.1, 3.5, 1.4, 0.2, 0.0], [4.9, 3.0, 1.4, 0.2, 0.0], [4.7, 3.2, 1.3, 0.2, 0.0], [4.6, 3.1, 1.5, 0.2, 0.0]]) \
            and sorted(ex2) == sorted([[7.0, 3.2, 4.7, 1.4, 1.0], [6.4, 3.2, 4.5, 1.5, 1.0], [6.9, 3.1, 4.9, 1.5, 1.0], [5.5, 2.3, 4.0, 1.3, 1.0]]))
        print("test_split_examples passed")
   
    def test_predict(self):
        data1 = read_data("iris_reduced1.csv")
        root = init_tree(data1)
        full_tree = learn_dt(root, data1) # note: add "features" parameter if necessary
        d1 = predict(full_tree, [1,0,0,0,0])
        self.assertTrue(d1 == 0)
        data2 = read_data("iris_reduced2.csv")
        root2 = init_tree(data2)
        full_tree1 = learn_dt(root2, data2)
        d2 = predict(full_tree1, [1,0,0,0,0])
        self.assertTrue(d2 == 0)
        d3 = predict(full_tree1, [5.4, 2.6, 0, 1])
        self.assertTrue(d3 == 1)
        print("test_predict passed")

if __name__ == '__main__':
    unittest.main()
