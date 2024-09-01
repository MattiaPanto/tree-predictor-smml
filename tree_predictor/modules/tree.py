import numpy as np
import pandas as pd
import math
import concurrent.futures
import random
from modules.ML_tools import zero_one_loss

class _Node:
    def __init__(self, decision_criterion = None, left_child = None, right_child = None, is_leaf = False, label=None, info = None):
        self.decision_criterion = decision_criterion
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.label = label

        self.info = info

    def make_decision(self, data_point):
        feature = self.decision_criterion[0]
        is_cat = self.decision_criterion[2]
        if is_cat:
            category = self.decision_criterion[1]
            res = data_point[feature] == category
        else:
            threshold = self.decision_criterion[1]
            res = data_point[feature] < threshold
        return res
    
        

class TreePredictor:
    def __init__(self, max_depth = 20, min_samples_leaf = 2, splitting_criterion = "gini", num_candidate_attributes = None, random_state = None):
        self.root = None
        self.feature_thresholds = None
        self.total_mistakes = 0

        if splitting_criterion not in ['gini', 'scaled_entropy', 'sqrt_impurity']:
            raise ValueError("splitting_criterion must be 'gini' or 'entropy' or 'sqrt_impurity'")
        
        self.splitting_criterion = splitting_criterion   
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_candidate_attributes = num_candidate_attributes
        self.random_gen = random.Random(random_state)

        self.training_error = None
        self.tree_info = {
            "num_leaves" : 0,
            "max_depth" : 0,
        }

        
        
    
    def train(self, X, y, num_thresholds = None):
        """
        Trains the tree predictor.

        Args:
            X (pd.DataFrame): Input features for training
            y (pd.DataFrame): Output labels
            num_thresholds (int): Number of thresholds for numerical values

        Returns:
            float: training error
        """
        self.total_mistakes = 0
        self.feature_thresholds = self._get_possible_thresholds(X, num_thresholds)
        
        # Train the predictor
        self.root = self._grow_tree(X, y)
        self.training_error = self.total_mistakes / y.size

    def predict(self, X):   
        predictions = []
        for i, data_point in X.iterrows():  
            label = self.predict_datapoint(data_point)
            predictions.append(label)
        return predictions

    def predict_datapoint(self, x):
        if self.root is None:
            raise ValueError("Predictor not trained")
        
        node = self.root
        while not node.is_leaf:
            if node.make_decision(x) == True:
                node = node.left_child
            else:
                node = node.right_child
        return node.label
    
    def gini_index(self, p):
        return 2 * p * (1 - p)
    
    def scaled_entropy(self, p):
        return 0 if (p == 0 or p == 1) else (-p/2 * math.log2(p)) - ((1-p)/2 * math.log2(1-p)) 
    
    def sqrt_impurity(self, p):
        return math.sqrt(p*(1-p))

    def _grow_tree(self, X, y, curr_depth=0):
        # Check the max_depth stopping criterion
        if self._pre_stopping_criterion(y, curr_depth):
            return self._get_leaf_node(y,curr_depth)         

        # Find the best split
        best_criterion, left_indices, right_indices, info = self._find_best_split(X, y)

        # Check the others stopping criterion
        if self._post_stopping_criterion(left_indices, right_indices):      
            return self._get_leaf_node(y,curr_depth) 
        
        left_subtree = self._grow_tree(X.loc[left_indices], y.loc[left_indices], curr_depth + 1)
        right_subtree = self._grow_tree(X.loc[right_indices], y.loc[right_indices], curr_depth + 1)
        
        return _Node(decision_criterion=best_criterion, left_child=left_subtree, right_child=right_subtree, info = info)

    def _get_leaf_node(self, y, curr_depth):
        leaf_label = self._compute_leaf_label(y=y)  
        self.total_mistakes += y.size - y.value_counts().get(leaf_label, 0) 

        # tree info
        self.tree_info["num_leaves"] += 1
        if self.tree_info["max_depth"] < curr_depth: 
            self.tree_info["max_depth"] = curr_depth

        # leaf info
        counts = y['class'].value_counts()
        node_info = {
            "num_samples" : y.size,
            "True_samples" : counts.get(True, 0),
            "False_samples" : counts.get(False, 0),
            "label" : leaf_label
        }

        return _Node(is_leaf=True, label=leaf_label, info = node_info) 

    def _compute_leaf_label(self, y):
        return y.mode()['class'][0]
    
    def _get_possible_thresholds(self, X, num_thresholds = None):
        """
        Given a dataset, this function returns a dictionary with one entry for each feature, where each entry contains
        an array of possible thresholds and a boolean value that is true if the feature is categorical, false if it is continuous.
        """
        feature_thresholds = dict()
        features = X.columns.tolist()
        
        for f in features:
            col = X[f]
            if pd.api.types.is_numeric_dtype(col):
                # La feature è numerica
                sorted_values = np.sort(col)
                thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2.0

                if num_thresholds is not None:
                    step = math.ceil(X.shape[0] / num_thresholds)
                    temp_thresholds = thresholds[::step]
                    if len(temp_thresholds) < num_thresholds:
                        np.append(temp_thresholds, thresholds[-1])
                    thresholds = temp_thresholds

                categorical = False

            else:
                thresholds = col.dropna().unique().tolist()
                categorical = True
            
            feature_thresholds[f] = (thresholds, categorical)

        return feature_thresholds
     
    def _find_best_split(self, X, y):
        node_info = {
            "best_feature" : None,
            "best_threshold" : None,
            "impurity" : None,
            "num_samples" : None,
            "max_gain" : None
        } 

        # Compute the initial impurity
        p = y.value_counts().get(True, 0) / y.size

        max_gain = 0
        best_feature = None
        best_category = None
        best_threshold = None

        best_left_indices = None
        best_right_indices = None
        is_best_cat = None

        # Sample a subset of features
        blocked_features = []
        if self.num_candidate_attributes is not None:
            features = list(self.feature_thresholds.keys())

            if self.num_candidate_attributes > len(features):
                raise ValueError(f"num_candidate_attributes ({self.num_candidate_attributes}) cannot be greater than the number of features ({len(features)}).")

            blocked_features  = self.random_gen.sample(features, len(features) - self.num_candidate_attributes)

        for feature, (possible_thresholds, is_cat) in self.feature_thresholds.items():
            # Skip feature if it is in blocked_features
            if feature in blocked_features: 
                continue

            #Explore all thresholds or categories for the features
            for threshold in possible_thresholds:

                #Compute indices for left and right
                if is_cat:
                    left_indices = X.index[X[feature] == threshold].tolist()
                    right_indices = X.index[(X[feature] != threshold)].tolist()
                else: 
                    left_indices = X.index[X[feature] < threshold].tolist()
                    right_indices = X.index[X[feature] >= threshold].tolist()

                left_labels = y.loc[left_indices]
                right_labels = y.loc[right_indices]

                
                #Skip if empty
                if left_labels.size == 0 or right_labels.size == 0:
                    continue
                
                # COmpute the impurity measure for left and right branches
                q = left_labels.value_counts().get(True, 0) / (left_labels.size)
                r = right_labels.value_counts().get(True, 0) / (right_labels.size)
                alpha = left_labels.size / y.size

                #Compute the gain
                if self.splitting_criterion == 'gini':
                    pre_split_impurity = self.gini_index(p)
                    left_impurity = self.gini_index(q)
                    right_impurity = self.gini_index(r)

                elif self.splitting_criterion == 'scaled_entropy':
                    pre_split_impurity = self.scaled_entropy(p)
                    left_impurity = self.scaled_entropy(q)
                    right_impurity = self.scaled_entropy(r)

                elif self.splitting_criterion == 'sqrt_impurity':
                    pre_split_impurity = self.sqrt_impurity(p)
                    left_impurity = self.sqrt_impurity(q)
                    right_impurity = self.sqrt_impurity(r)
                else:
                    raise ValueError("splitting_criterion must be 'gini' or 'entropy' or 'sqrt_impurity'")
                
                gain = pre_split_impurity - (alpha * left_impurity + (1 - alpha) * right_impurity)

                #Update the best split if the current gain is higher
                if gain > max_gain:
                    max_gain = gain
                    best_feature = feature
                    best_left_indices = left_indices
                    best_right_indices = right_indices
                    if is_cat:
                        best_category = threshold
                        is_best_cat = True
                    else:
                        best_threshold = threshold
                        is_best_cat = False

                    node_info["best_feature"] = feature
                    node_info["best_threshold"] = threshold
                    node_info["impurity"] = pre_split_impurity
                    node_info["num_samples"] = X.shape[0]
                    node_info["max_gain"] = max_gain
                    

        if is_best_cat:
            return (best_feature, best_category, is_best_cat), best_left_indices, best_right_indices, node_info
        elif not is_best_cat:
            return (best_feature, best_threshold, is_best_cat), best_left_indices, best_right_indices, node_info
        else: 
            return None, None, None
    
    def _pre_stopping_criterion(self, y, curr_depth):  
        if self.max_depth is not None and curr_depth >= self.max_depth: 
            return True

        unique_class = y['class'].dropna().unique()
        if len(unique_class) == 1: 
            return True
             
        return False

    def _post_stopping_criterion(self, left_indices, right_indices):
        if left_indices is None or right_indices is None:           
            return True 
        
        if self.min_samples_leaf is not None and (len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf):
            return True
        return False
    


class RandomForest():
    def __init__(self, max_depth = 20, min_samples_leaf = 1, splitting_criterion = "gini", num_candidate_attributes = None, compute_oob_score = False, random_state = 1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.splitting_criterion = splitting_criterion
        self.num_candidate_attributes = num_candidate_attributes
        self.compute_oob_score = compute_oob_score
        self.random_gen = random.Random(random_state)

        self.training_error = None
        self.oob_error = None

        self._random_forest = list()
        self._oob_indices_list = list()
        


    def _train_tree_predictor(self, X, y, num_thresholds = 1, random_state = None):
        #Sample with replacement
        dataset = pd.concat([X, y], axis=1)

        bootstrap_sample = dataset.sample(frac = 1, replace = True, random_state=random_state) 
        oob_indices = list(set(range(dataset.shape[0])) - set(bootstrap_sample.index))

        # Reset indices
        bootstrap_sample = bootstrap_sample.reset_index(drop=True)

        new_X = dataset.drop(columns=['class'])
        new_y = dataset[['class']]

        tp = TreePredictor(self.max_depth, self.min_samples_leaf, self.splitting_criterion, self.num_candidate_attributes, random_state)
        tp.train(new_X, new_y, num_thresholds=num_thresholds)
        training_error = tp.training_error
        return tp, oob_indices, training_error


    def train(self, X, y, num_thresholds=1, num_trees=3, max_workers=None):
        """
        Trains the random forest predictor.

        Args:
            X (pd.DataFrame): Input features for training
            y (pd.DataFrame): Output labels
            num_thresholds (int): Number of thresholds to consider for numerical splits
            num_trees (int)
            max_workers (int or None): Maximum number of workers for parallel tree training. If None, trees are trained sequentially
        Returns:
            float: The average training error across all trained trees
        """
        self.random_forest = []
        self.oob_indices_list = []
        random_state_list = [self.random_gen.randint(0, 10000) for _ in range(num_trees)]

        results = []
        counter = 1
        if max_workers is None:
            for i in range(num_trees):
                results.append(self._train_tree_predictor(X, y, num_thresholds))
                print(counter, "/", num_trees)
                counter += 1
        else: 
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._train_tree_predictor, X, y, num_thresholds, random_state_list[id]) for id in range(num_trees)]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                        print(counter, "/", num_trees, end="\r")
                        counter += 1
                    except Exception as e:
                        print(f"An error occurred: {e}")

        training_error_list = []
        for (tree, oob_indices, training_error) in results:
            self._random_forest.append(tree)
            self._oob_indices_list.append(oob_indices)
            training_error_list.append(training_error)

        self.mean_training_error = np.mean(training_error_list)
        if self.compute_oob_score is True:
            self.oob_error = self._compute_oob_error(X, y)

    
    def predict(self, X, tree_indices = None):   
        predictions = []
        for i, data_point in X.iterrows():  
            label = self.predict_datapoint(data_point, tree_indices)
            predictions.append(label)
        return predictions

    def predict_datapoint(self, x, tree_indices = None):
        if len(self._random_forest) == 0:
            raise ValueError("Predictor not trained")
        
        true_count = 0
        false_count = 0

        if tree_indices is None: 
            tree_indices = range(0, len(self._random_forest))
            
        for predictor in np.array(self._random_forest)[tree_indices]:
            label = predictor.predict_datapoint(x)
            if label:
                true_count += 1
            else:
                false_count +=1
        
        if true_count > false_count: return True
        else: return False
        
    def _compute_oob_error(self, X, y):
        prediction_list = []
        true_label_list = []

        for i, data_point in X.iterrows():
            tree_indices = []
            for tree_index, oob in enumerate(self._oob_indices_list):
                if i in oob:
                    tree_indices.append(tree_index)

            if len(tree_indices) > 0:
                prediction = self.predict_datapoint(data_point, tree_indices)
                prediction_list.append(prediction)
                true_label_list.append(y["class"][i])

        if len(true_label_list) > 0:
            oob_error = zero_one_loss(true_label_list, prediction_list)
            return oob_error
        else:
            return None

                    