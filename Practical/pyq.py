import numpy as np

def best_split(features, targets):
    '''
    inputs:
        features: nd-array
        targets: nd-array
    output:
        integer value determining best attribute idx (1-based indexing) for decision tree regression
    '''

    best_feature_idx = None
    best_value = None
    mse_base = 1e9  # Initialize with a very high MSE

    # Iterating through each of the features
    for feature_idx in range(features.shape[1]):
        # Observations in left and right of the node
        left_y = targets[features[:, feature_idx] == 0]  # Left node: target values where feature is 0
        right_y = targets[features[:, feature_idx] == 1] # Right node: target values where feature is 1

        # Calculate the means 
        left_mean = np.mean(left_y) if len(left_y) > 0 else 0
        right_mean = np.mean(right_y) if len(right_y) > 0 else 0

        # Calculate the left and right residuals 
        res_left = left_y - left_mean
        res_right = right_y - right_mean

        # Calculate the MSE
        mse_split = np.sum(res_left * 2) + np.sum(res_right * 2)

        # Checking if this is the best split so far 
        if mse_split < mse_base:
            best_feature_idx = feature_idx
            mse_base = mse_split  # Update the best (lowest) MSE

    return best_feature_idx + 1  # For 1-based indexing


import numpy as np
np.random.seed(0)

x = eval(input())  # Replace with actual input
y = eval(input())  # Replace with actual input

np.random.seed(0)

def adab_predict(x, alpha, clfs):
    
    # clf_preds consists of alpha[i] * prediction for each observation by each classifier
    clf_preds = np.asarray([alpha[i] * clfs[i].predict(x) for i in range(len(clfs))])

    # y_pred represents the weighted sum of predictions from all classifiers for each observation
    y_pred = np.sum(clf_preds, axis=0)
    
    # Convert the weighted sum into class labels (+1 or -1) based on the sign
    y_pred = np.sign(y_pred)

    return y_pred.astype('int32')