import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from itertools import product
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def zero_one_loss(y_true, y_pred):
    set_size = len(y_true)
    if set_size != len(y_pred):
        raise ValueError(f"The number of true labels ({len(y_true)}) does not match the number of predictions ({len(y_pred)})")
    
    total_mistakes = 0
    for i in range(set_size):
        if y_true[i] != y_pred[i]: 
            total_mistakes += 1
    
    return total_mistakes / set_size

def precision(y_true, y_pred):
    tp = sum((y_true[i] == 1) and (y_pred[i] == 1) for i in range(len(y_true)))
    fp = sum((y_true[i] == 0) and (y_pred[i] == 1) for i in range(len(y_true)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return precision

def recall(y_true, y_pred):
    tp = sum((y_true[i] == 1) and (y_pred[i] == 1) for i in range(len(y_true)))
    fn = sum((y_true[i] == 1) and (y_pred[i] == 0) for i in range(len(y_true)))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return recall

def F1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if (p + r) == 0:
        return 0
    return 2 * (p * r) / (p + r)


def CV_score(predictor, X, y, num_thresholds, k):
    """
    Performs k-fold cross-validation.

    Args:
        predictor: The model object
        X (pd.DataFrame): Input features
        y (pd.DataFrame): Output labels
        num_thresholds (int): Number of thresholds for numerical values
        k (int): Number of folds for cross-validation

    Returns:
        tuple: A tuple containing the average accuracy, precision, recall, and F1-score across all folds
               - float: mean accuracy
               - float: mean precision
               - float: mean recall
               - float: mean F1_score
    """    
    kf = KFold(n_splits=k)

    accuracy_list = []
    precision_list = []
    recall_list = []
    F1_score_list = []

    num_fold = 1
    print(f"", end="\r")
    for train_index, validation_index in kf.split(X):
        print(f"fold {num_fold}/{k}")
        X_train, X_val = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_val = y.iloc[train_index], y.iloc[validation_index]
        
        imputer = MissingValuesImputer()
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val) 

        predictor.train(X_train, y_train, num_thresholds = num_thresholds)  
        y_pred = predictor.predict(X_val) 
        
        accuracy_list.append(1-zero_one_loss(y_val['class'].tolist(), y_pred))
        precision_list.append(precision(y_val['class'].tolist(), y_pred))
        recall_list.append(recall(y_val['class'].tolist(), y_pred))
        F1_score_list.append(F1_score(y_val['class'].tolist(), y_pred))

        num_fold += 1

    
    return np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(F1_score_list)



def GridSearchCV(predictor_class, X, y, num_thresholds, k, param_grid):
    """
    Performs a grid search with cross-validation for a predictor.

    Args:
        predictor_class: The class of the predictor to be evaluated
        X (pd.DataFrame): Input features
        y (pd.DataFrame): Output labels
        num_thresholds (int): Number of thresholds for numerical values
        k (int): Number of folds for cross-validation
        param_grid (dict): Dictionary where keys are the names of hyperparameters and values are lists of values to try

    Returns:
        list: A list of tuples, where each tuple contains:
              - dict: The combination of hyperparameters used
              - float: Mean accuracy
              - float: Mean precision
              - float: Mean recall
              - float: Mean F1_score
    """
    param = param_grid.keys()
    values = param_grid.values()

    combinations = list(product(*values))
    print("Num combinations", len(combinations))

    param_list = []
    for i, comb in enumerate(combinations):
        comb_dict = dict(zip(param, comb))
        print(f"{i}  {comb_dict}")

        p = predictor_class(**comb_dict) # passa i parametri del dizionario come attributo   
        # Cross-validation
        accuracy, pr, re, f1 = CV_score(p, X, y, num_thresholds, k=k)

        print(f"validation accuracy: {accuracy}")
        print(f"validation precision: {pr}")
        print(f"validation recall: {re}")
        print(f"validation f1 {f1}")
        print("----------------")

        param_list.append((comb_dict, accuracy, pr, re, f1))
        

    return param_list

def compute_mutual_information(X, y, random_state = 1):
    categorical_indices = []
    for i, col in enumerate(X.columns):
        if X[col].dtype.name == 'object' or X[col].dtype.name == 'category':
            categorical_indices.append(i)

    x_enc = X.apply(LabelEncoder().fit_transform)
    mutual_info = mutual_info_classif(x_enc, y['class'].tolist(), discrete_features=categorical_indices, random_state=random_state)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = x_enc.columns
    mutual_info.sort_values(ascending=False, inplace=True)
    return mutual_info




class MissingValuesImputer:
    def __init__(self):
        self.data_replacement = dict()

    def fit_transform(self, X):
        X_copy = X.copy()
    
        for column in X_copy.columns:
            if pd.api.types.is_numeric_dtype(X_copy[column]):
                # Sostituisce con media per attributi numerali
                mean_value = X_copy[column].mean()
                self.data_replacement[column] = mean_value
                X_copy.fillna({column: mean_value}, inplace=True)
            else:
                # Sostituisce con moda per attributi categorici
                mode_value = X_copy[column].mode()[0]
                self.data_replacement[column] = mode_value
                X_copy.fillna({column: mode_value}, inplace=True)
        
        return X_copy


    def transform(self, X):       
        X_copy = X.copy()

        for column in X_copy.columns:
            X_copy.fillna({column: self.data_replacement[column]}, inplace=True)
        
        return X_copy
    

