import MDP_function as mf
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import scipy.stats as stats
import random as rnd

# read in all original data set
data_file = "MDP_Original_data.csv"
original_data = pd.read_csv(data_file)

# select all features from the original dataset and get features names as feature space indexes
feature_data = original_data.loc[:, 'Interaction':'CurrPro_medianProbTime']
feature_space = feature_data.columns.tolist()

# initialize parameters and data structures for correlation-based feature selection algorithm
MAX_NUM_OF_FEATURES = 8
ECR_list = list()
optimal_feature_set = list()

def feature_discretization(feature_data, maxLevel=2): 
    # discretize continuous feature values into integers of no more than max levels
    isFloat = any(map(lambda x: isinstance(x, float), feature_data)) # check if it contain float type
    if not isFloat:
        isOverLevel = len(feature_data.unique())>maxLevel # check if it is within max levels
    if isFloat or isOverLevel: # discretize and reduce levels using median
        median = feature_data.median()
        feature_vals = map(lambda x: 0 if x<=median else 1, feature_data)
        feature_data = pd.Series(feature_vals, dtype=int)
    return feature_data

    # discretization feature values by median
all_data_discretized = original_data.loc[:, "student":"reward"]
for i, ft in enumerate(feature_space):
    ft_data = original_data.loc[:, ft]
    all_data_discretized[ft] = feature_discretization(ft_data)
#all_data_discretized.describe()

# initialization to find the best feature with max ECR
for i, ft in enumerate(feature_space):
    #print i
    selected_feature = [ft]
    # try:
    ECR_list.append(mf.compute_ECR(all_data_discretized, selected_feature))
#     except:
#         ECR_list.append(0.0)

optimal_feature_set.append(feature_space[ECR_list.index(max(ECR_list))])

max(ECR_list)

def compute_correlation(dataset, feature_set, feature):
    corr_sum = 0
    for ft in feature_set:
        corr, p_val = stats.pearsonr(dataset[ft], dataset[feature])
        corr_sum += corr
    return corr_sum
    # return rnd.random()

# feature selection iterations
while (len(optimal_feature_set) < MAX_NUM_OF_FEATURES):
    #print len(optimal_feature_set)
    corr_list = list()
    remain_feature_space = list(set(feature_space) - set(optimal_feature_set))
    for i,ft in enumerate(remain_feature_space):
        corr_list.append([ft, compute_correlation(all_data_discretized, optimal_feature_set, ft)])
    topK = 5
    top_features = map(lambda x: x[0], sorted(corr_list, key=lambda x: x[1], reverse=False)[:topK])
    ECR_list = list()
    for i, ft in enumerate(top_features):
        select_feature = list(optimal_feature_set)
        select_feature.append(ft)
        ECR_list.append([ft, mf.compute_ECR(all_data_discretized, selected_feature)])
        print ECR_list
    best_next_feature_and_ECR = sorted(ECR_list, key=lambda x: x[1], reverse=True)[0]
    best_next_feature, bestECR = best_next_feature_and_ECR[0], best_next_feature_and_ECR[1]
    optimal_feature_set.append(best_next_feature)
    print best_next_feature_and_ECR

print optimal_feature_set

mf.induce_policy_MDP2(all_data_discretized, optimal_feature_set)