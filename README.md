# Feature Selection For Reinforcement Learning In Educational Policy Development


This project aim to develop feature selection method to improve the overall ECR(Expected Cumulative Reward) value in a recently [published work](http://dl.acm.org/citation.cfm?id=2930247), which studied policies to improve students' learning (measured by ECR) using reinforcement learning model. The  reinforcement learning system built a Markov model consisting of tutor actions, learning context (features) as states and student learning as reward. We develop a feature selection model significantly improving the previous work in the publication using a forward-backward greedy algorithm of a wrapper feature selection method.

![Proposed Algorithm](https://github.com/jerry-shijieli/CSC591_AssignedProject_Feature_Selection_For_Reinforcement_Learning/blob/master/image/feature_selection_algorithm.png)

### Prerequisites

Following Python packages have to be installed before executing the project code

```
pymdptoolbox
numpy
scipy
pandas
sklearn
nltk==3.1
matplotlib
seaborn
progressbar
```
* Modification on pymdptoolbox installed module codes is needed to make it work in this project. See the following instructions.

## Running the tests

All codes are in _code/_ folder.

Our feature discretization and selection codes are in the file named: 
```
    mdp_feature_selection_greedy_forward_backward.py
```

To run this code, type command in terminal as:
```
    $python mdp_feature_selection_greedy_forward_backward.py
```

The code will read input file MDP_Original_data2.csv, compute ECR to select best feature set and print out best policy.

Since this code may takes a long time (about 3 hours), we rewrite this code into a python notebook:
```
    mdp_feature_selection_greedy_forward_backward.ipynb
```
    
and executed this notebook. So all the intermediate and final results have been recorded in this notebook, you can view it through web browser. To open this notebook in web browser, type command in terminal as:
```
    $ipython notebook
```
    
This notebook depends on the given code for ECR and policy computation: 
```
    MDP_function.py
```
    
Also we modified the code in mdp.py to avoid policy reward computation issue. Please replace original mdp.py by our code before running other code files.

Please feel free to contact us if you have any question concerning this project! 

## Expected Results

The result is to improve the overall ECR using selected features and generate cooresponding policies in the state space (represented by binary feature value combination). Our best result achieved overall ECR above 350 with a policy coverage around 89%, which are both significant improvement over the original work. 

![Policy Visualization](https://github.com/jerry-shijieli/CSC591_AssignedProject_Feature_Selection_For_Reinforcement_Learning/blob/master/image/policy_visualization.png)

In addition, we also prove that the feature-correlation-based feature selection method does not work well by the plot of correlation matrix between single action reward and optimal features.

![Optimal feature correlation matrix with reward included](https://github.com/jerry-shijieli/CSC591_AssignedProject_Feature_Selection_For_Reinforcement_Learning/blob/master/image/correlation_of_optimal_feature_vs_reward.png)

## Contributors

* **Shijie Li**  *(email: sli41@ncsu.edu)* 
* **Cong Mai** *(email: cmai@ncsu.edu)*
* **Yifan Guo** *(email: yguo14@ncsu.edu)*

## Acknowledgments

* Thank Prof. Min Chi for the support on this project.
* Thank all TAs of CSC591 course for the evaluation and feedback on this project.


