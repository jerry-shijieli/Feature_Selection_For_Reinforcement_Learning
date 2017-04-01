import MDP_function as mf
import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
import scipy.stats as stats
import random as rnd

# read in all original data set
data_file = "MDP_Original_data2.csv"
original_data = pd.read_csv(data_file)

# select all features from the original dataset and get features names as feature space indexes
feature_data = original_data.loc[:, 'Interaction':'CurrPro_medianProbTime']
feature_space = feature_data.columns.tolist()

# helper functions

def feature_discretization_by_median(feature_data, maxLevel=2): 
    # discretize continuous feature values into integers of no more than max levels
    isFloat = any(map(lambda x: isinstance(x, float), feature_data)) # check if it contain float type
    if not isFloat:
        isOverLevel = len(feature_data.unique())>maxLevel # check if it is within max levels
    if isFloat or isOverLevel: # discretize and reduce levels using median
        median = feature_data.median()
        feature_vals = map(lambda x: 0 if x<=median else 1, feature_data)
        feature_data = pd.Series(feature_vals, dtype=int)
    return feature_data

def compute_correlation(dataset, feature_set, feature):
    # corr_sum = 0
    # for ft in feature_set:
    #     corr, p_val = stats.pearsonr(dataset[ft], dataset[feature])
    #     corr_sum += corr
    # return corr_sum
    return rnd.random()

# initialize parameters and data structures for correlation-based feature selection algorithm
MAX_NUM_OF_FEATURES = 8
ECR_list_of_single_feature = list()
optimal_feature_set = list()
max_total_ECR = 0

# discretization feature values by median
all_data_discretized = original_data.loc[:, "student":"reward"]
for i, ft in enumerate(feature_space):
    ft_data = original_data.loc[:, ft]
    all_data_discretized[ft] = feature_discretization_by_median(ft_data)
all_data_discretized = pd.DataFrame(all_data_discretized)

# initialization to find the best feature with max ECR
for ft in feature_space:
    selected_feature = [ft]
    ECR_list_of_single_feature.append(mf.compute_ECR(all_data_discretized, selected_feature))

# initialize the optimal feature set with feature of highest ECR
max_total_ECR = max(ECR_list_of_single_feature)
optimal_feature_set.append(feature_space[ECR_list_of_single_feature.index(max_total_ECR)])
print(optimal_feature_set)

# feature selection iterations
correlation_rank_reverse = False
while (len(optimal_feature_set) < MAX_NUM_OF_FEATURES):
    print "####### Search next feature on iteration: "+str(len(optimal_feature_set))+" #######"
    remain_feature_space = list(set(feature_space) - set(optimal_feature_set)) # features not in optimal feature set
    # feature selection heuristics
    corr_list = list() # correlation between new feature and optimal feature set
    for i,ft in enumerate(remain_feature_space):
        corr_list.append([ft, compute_correlation(all_data_discretized, optimal_feature_set, ft)])
    topK = 5+int(0.03*np.exp(len(optimal_feature_set))) # dynamically choose top-K candidate features based on feature similarity metrics
    top_features = map(lambda x: x[0], sorted(corr_list, key=lambda x: x[1], reverse=correlation_rank_reverse)[:topK])
    # select optimal feature from candidate set based on ECR value
    ECR_list = list() # ECR values of optimal feature set with new candidate feature
    for ft in top_features:
        selected_feature = list(optimal_feature_set)
        selected_feature.append(ft) # combine candidate feature to optimal feature set
        ECR_with_ft_added = mf.compute_ECR(all_data_discretized, selected_feature)
        print "Candidate feature: "+ ft +" --> ECR value:"+ str(ECR_with_ft_added)
        if (ECR_with_ft_added > max_total_ECR):
            print "Qualified candidate feature added +"
            ECR_list.append([ft, ECR_with_ft_added])
        else:
            print "Unqualified candidate feature skipped -"
    if (not ECR_list): # if no new qualified candidate feature, keep searching
    	correlation_rank_reverse = not correlation_rank_reverse
        continue
    else:
        best_next_feature, bestECR = sorted(ECR_list, key=lambda x: x[1], reverse=True)[0]
        optimal_feature_set.append(best_next_feature)
        max_total_ECR = bestECR
    # check potential for improving ECR over all subsets of optimal feature set
    size_of_optimal_feature_set = len(optimal_feature_set)
    if (size_of_optimal_feature_set >= MAX_NUM_OF_FEATURES): 
    	while (size_of_optimal_feature_set > 1):
	        ECR_list = map(lambda ft_index: mf.compute_ECR(all_data_discretized, 
	                        optimal_feature_set[:ft_index]+optimal_feature_set[ft_index+1:]), 
	                        range(size_of_optimal_feature_set-1)) # calculate ECR for optimal feature subsets
	        max_ECR_in_subset = max(ECR_list)
	        if (max_ECR_in_subset >= max_total_ECR): # choose subset with ECR no less than highest overall ECR
	            print "Better optimal feature subset is discovered!"
	            ft_index_of_max_subset_ECR = ECR_list.index(max_ECR_in_subset)
	            optimal_feature_set = optimal_feature_set.pop(ft_index_of_max_subset_ECR)
	            max_total_ECR = max_ECR_in_subset
	            size_of_optimal_feature_set = len(optimal_feature_set)
    # keep record of the highest ECR and its optimal feature set so far
    print "Highest ECR so far: "+str(max_total_ECR)+" with optimal feature set as:"
    print optimal_feature_set

mf.induce_policy_MDP2(all_data_discretized, optimal_feature_set)