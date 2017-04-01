import collections
import numpy as np
import pandas as pd
import mdptoolbox, mdptoolbox.example
import argparse
import scipy.stats as stats
import random as rnd

def generate_MDP_input2(original_data, features):

    students_variables = ['student', 'priorTutorAction', 'reward']

    # generate distinct state based on feature
    if (type(original_data[features])==pd.DataFrame):
        original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1) # pd.DataFrame
    else:
        original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x)) # pd.Series
    # original_data['state'] = original_data[features].apply(tuple, axis=1)
    students_variables = students_variables + ['state']
    data = original_data[students_variables]

    # quantify actions
    distinct_acts = list(data['priorTutorAction'].unique())
    Nx = len(distinct_acts)
    i = 0
    for act in distinct_acts:
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1

    # initialize state transition table, expected reward table, starting state table
    # distinct_states didn't contain terminal state
    student_list = list(data['student'].unique())
    distinct_states = list()
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        # don't consider last row
        temp_states = list(student_data['state'])[0:-1]
        distinct_states = distinct_states + temp_states
    distinct_states = list(set(distinct_states))

    Ns = len(distinct_states)

    # we include terminal state
    start_states = np.zeros(Ns + 1)
    A = np.zeros((Nx, Ns+1, Ns+1))
    expectR = np.zeros((Nx, Ns+1, Ns+1))

    # update table values episode by episode
    # each episode is a student data set
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        # count the number of transition among states without terminal state
        for i in range(1, (len(row_list)-1)):
            state1 = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

        # count the number of transition from state to terminal
        state1 = distinct_states.index(student_data.loc[row_list[-2], 'state'])
        act = student_data.loc[row_list[-1], 'priorTutorAction']
        A[act, state1, Ns] += 1
        expectR[act, state1, Ns] += float(student_data.loc[row_list[-1], 'reward'])

    # normalization
    start_states = start_states / np.sum(start_states)

    for act in range(Nx):
        A[act, Ns, Ns] = 1
        # generate expected reward
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        # some states only have either PS or WE transition to other state
        for l in np.where(np.sum(A[act], axis=1) == 0)[0]:
            A[act, l, l] = 1
            
        A[act] = np.divide(A[act].transpose(), np.sum(A[act], axis=1))
        A[act] = A[act].transpose()

    return [start_states, A, expectR, distinct_acts, distinct_states]


def calcuate_ECR(start_states, expectV):
    ECR_value = start_states.dot(np.array(expectV))
    return ECR_value


def output_policy(distinct_acts, distinct_states, vi):
    Ns = len(distinct_states)
    print('Policy: ')
    print('state -> action, value-function')
    for s in range(Ns):
        print(distinct_states[s] + " -> " + distinct_acts[vi.policy[s]] + ", " + str(vi.V[s]))

def induce_policy_MDP2(original_data, selected_features):

    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input2(original_data, selected_features)

    # apply Value Iteration to run the MDP
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
    vi.run()

    # output policy
    output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    print('ECR value: ' + str(ECR_value))
    return ECR_value

def compute_ECR(original_data, selected_features):

    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input2(original_data, selected_features)

    # apply Value Iteration to run the MDP
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
    vi.run()

    # output policy
    #output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    #print('ECR value: ' + str(ECR_value))
    return ECR_value

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



if __name__ == "__main__":

    # read in all original data set
    data_file = "MDP_Original_data2.csv"
    original_data = pd.read_csv(data_file)

    # select all features from the original dataset and get features names as feature space indexes
    feature_data = original_data.loc[:, 'Interaction':'CurrPro_medianProbTime']
    feature_space = feature_data.columns.tolist()

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
        ECR_list_of_single_feature.append(compute_ECR(all_data_discretized, selected_feature))

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
            ECR_with_ft_added = compute_ECR(all_data_discretized, selected_feature)
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
            is_subset_better = True
            while (size_of_optimal_feature_set>1 and is_subset_better):
                ECR_list = map(lambda ft_index: compute_ECR(all_data_discretized, 
                                optimal_feature_set[:ft_index]+optimal_feature_set[ft_index+1:]), 
                                range(size_of_optimal_feature_set-1)) # calculate ECR for optimal feature subsets
                max_ECR_in_subset = max(ECR_list)
                if (max_ECR_in_subset >= max_total_ECR): # choose subset with ECR no less than highest overall ECR
                    print "Better optimal feature subset is discovered!"
                    ft_index_of_max_subset_ECR = ECR_list.index(max_ECR_in_subset)
                    optimal_feature_set = optimal_feature_set.pop(ft_index_of_max_subset_ECR)
                    max_total_ECR = max_ECR_in_subset
                    size_of_optimal_feature_set = len(optimal_feature_set)
                else:
                    is_subset_better = False
        # keep record of the highest ECR and its optimal feature set so far
        print "Highest ECR so far: "+str(max_total_ECR)+" with optimal feature set as:"
        print optimal_feature_set

    induce_policy_MDP2(all_data_discretized, optimal_feature_set)