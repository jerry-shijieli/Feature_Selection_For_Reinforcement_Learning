# Feature Selection For Reinforcement Learning In Educational Policy Development

## Getting Started

The models and algorithms of our project is implemented in Python code with the help of IPython Notebook for data and result visualization. All experiments are executed on local machine.

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
* Note _nltk v3.2_ may have issue with stemming functions.

## Running the tests

All codes are in _code/_ folder.

Our feature discretization and selection codes are in the file named: 
    mdp_feature_selection_greedy_forward_backward.py

To run this code, type command in terminal as:
    $python mdp_feature_selection_greedy_forward_backward.py

The code will read input file MDP_Original_data2.csv, compute ECR to select best feature set and print out best policy.

Since this code may takes a long time (about 3 hours), we rewrite this code into a python notebook:
    mdp_feature_selection_greedy_forward_backward.ipynb
and executed this notebook. So all the intermediate and final results have been recorded in this notebook, you can view it through web browser. To open this notebook in web browser, type command in terminal as:
    $ipython notebook
This notebook depends on the given code for ECR and policy computation: 
    MDP_function.py
    
Also we modified the code in mdp.py to avoid policy reward computation issue. Please replace original mdp.py by our code before running other code files.

Please feel free to contact us if you have any question concerning this project! 

## Expected Results

The result is to improve the multi-class text classification accuracy by semi-supervised EM Naive Bayes classifier given both labeled and unlabeled documents.

![Classification Accuracy Improvement](https://github.com/jerry-shijieli/Text_Classification_Using_EM_And_Semisupervied_Learning/blob/master/result/cv_f1.png)

![Word Cloud of Most Probable Keywords in Each Class](https://github.com/jerry-shijieli/Text_Classification_Using_EM_And_Semisupervied_Learning/blob/master/result/test_em_nb_wc.png)

For more details and intermediate results, please check the ipython notebooks in the folder [code](https://github.com/jerry-shijieli/Text_Classification_Using_EM_And_Semisupervied_Learning/tree/master/code)


## Contributors

* **Shijie Li**  *(email: sli41@ncsu.edu)* 
* **Cong Mai** *(email: cmai@ncsu.edu)*
* **Yifan Guo** *(email: yguo14@ncsu.edu)*

## Acknowledgments

* Thank Prof. Min Chi for the support on this project.
* Thank all TAs of CSC591 course for the evaluation and feedback on this project.


