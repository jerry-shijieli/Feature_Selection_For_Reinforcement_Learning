# Author: Cong Mai

import collections 
import numpy as np
import pandas as pd
import mdptoolbox, mdptoolbox.example
import scipy.stats as stats
import random as rnd
import progressbar as pgb
import time
import os
import sys
import pickle
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def generate_MDP_input2(original_data, features):

    students_variables = ['student', 'priorTutorAction', 'reward']

    # generate distinct state based on feature
    try:
        original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1) # pd.DataFrame
    except:
        original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x) if isinstance(x,collections.Iterable) else str(x)) # pd.Series
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
        feature_data = map(lambda x: 0 if x<=median else 1, feature_data)
    feature_data = pd.Series(feature_data, dtype=int)
    return feature_data

def feature_discretization_by_multilevels(feature_data, maxLevel=3):
    return 0
    
def compute_correlation(dataset, feature_set, feature):
    # corr_sum = 0
    # for ft in feature_set:
    #     corr, p_val = stats.pearsonr(dataset[ft], dataset[feature])
    #     corr_sum += corr
    # return corr_sum
    return rnd.random()

def save_optimal_feature_selection(dataset, optimal_feature_set, save_info):
    # with open('Training_data.csv', 'w') as fout:
    data_to_save = dataset.loc[:, "student":"reward"]
    for ft in optimal_feature_set:
        data_to_save[ft] = dataset.loc[:, ft]
    try:
        data_to_save.to_csv('Training_data.csv', sep=',', index=False)
        with open("bestECR.log",'a') as fout:
            for term in save_info.keys():
                fout.write(term+save_info[term]+'\n')
            # fout.write("Highest ECR so far: "+str(max_total_ECR)+" with optimal feature set as:")
            # fout.write(', '.join(optimal_feature_set))
            fout.write('\n')
    except:
        print "Failed to save results!!!"


def binListString(x,bitLength):
    outputList=[str(0)]*bitLength
    binList=[str(int(i)) for i in str(bin(x))[2:]]
    w=len(binList)
    for i in range(0,w):
        outputList[i+bitLength-w]=binList[i]
    return ':'.join(outputList)

def binFindNum(dis_state,selectList):
    binList=dis_state.split(":")
    k=0
    
    for i in range(len(selectList)):
        
        k=int(binList[selectList[i]])+k*2
        #print selectList[i],binList[selectList[i]],k
    return k
def visual_policy(original_data, selected_features,rowList=[0,1,2,3],colList=[4,5,6,7]):

    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input2(original_data, selected_features)

    # apply Value Iteration to run the MDP
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
    vi.run()

    Ns = len(distinct_states)
    nrows=len(rowList)
    ncols=len(colList)
    image = np.ones((2**nrows,2**ncols))
   
    row_labels = []
    col_labels = []
    for x in range(2**nrows):
        row_labels.append(binListString(x,nrows))
    for x in range(2**ncols):
        col_labels.append(binListString(x,ncols))
    countWE=0
    countPS=0
    for s in range(Ns):
        print(distinct_states[s] + " -> " + distinct_acts[vi.policy[s]] + ", " + str(vi.V[s]))
        
        rowIndex=binFindNum(distinct_states[s],rowList)
        colIndex=binFindNum(distinct_states[s],colList)
        if distinct_acts[vi.policy[s]]=="WE":
           image[rowIndex,colIndex]=2
           countWE=countWE+1
        else:
           image[rowIndex,colIndex]=0
           countPS=countPS+1
    countNoRule= (2**nrows)*(2**ncols)-Ns
    print "WE cases:",countWE
    print "PS cases:",countPS
    print "no rule cases:",countNoRule
    fig, ax = plt.subplots()
    i = ax.matshow(image, cmap=cm.gray,interpolation='none')
    plt.xticks(range(2**ncols), col_labels, rotation='vertical')
    plt.yticks(range(2**nrows), row_labels)
   
    
    ax.set_xticks(np.arange(-.5, 2**ncols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 2**nrows, 1), minor=True);

    
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.set_xlabel('Last 4 features f4:f5:f6:f7')
    ax.set_ylabel('First 4 features f0:f1:f2:f3',rotation='vertical')
    plt.show()




if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # disable unnecessary warning. default='warn'
    # read in all original data set
    data_file = "MDP_Original_data2.csv"
    original_data = pd.read_csv(data_file)

    # select all features from the original dataset and get features names as feature space indexes
    feature_data = original_data.loc[:, 'Interaction':'CurrPro_medianProbTime']
    
   
    feature_space = feature_data.columns.tolist()
    print feature_space
    # initialize parameters and data structures for correlation-based feature selection algorithm
    MAX_NUM_OF_FEATURES = 2
    ECR_list_of_single_feature = list()
    optimal_feature_set = list()
    max_total_ECR = 0

    # discretization feature values by median
    all_data_discretized = original_data.loc[:, "student":"reward"]
    print ">>> Feature discretization ... "
    bar = pgb.ProgressBar()
    for ft in bar(feature_space):
        ft_data = original_data.loc[:, ft]
        all_data_discretized[ft] = feature_discretization_by_median(ft_data)  
    optimal_feature_set=["difficultProblemCountSolved",'cumul_symbolicRepresentationCount',"cumul_AppRatio","cumul_MorphCount","CurrPro_avgProbTimeDeviationPS","cumul_FDActionCount","ruleScoreEQUIV",'cumul_TotalWETime']
    del original_data
    
    visual_policy(all_data_discretized, optimal_feature_set,rowList=[0,1,2,3],colList=[4,5,6,7])
