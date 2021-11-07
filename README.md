# Financial Scoring Model (#30)
This project is 1st selection of proposed projects from Neural Networks and Applications (XAI606) 2021 Fall.

**Table of Contents**
- [Financial Scoring Model (#30)](#financial-scoring-model-30)
  - [Project Description](#project-description)
    - [Dataset: HELOC](#dataset-heloc)
    - [Task & Goal](#task--goal)
  - [Proposed Methods](#proposed-methods)
    - [1. Improving Performance](#1-improving-performance)
    - [2. Finding features](#2-finding-features)
  - [Results](#results)
    - [Tree-based Classifiers](#tree-based-classifiers)
    - [TabNet](#tabnet)
  - [Conclusion](#conclusion)
  - [References](#references)

## Project Description

### Dataset: HELOC


**HELOC**, Home Equity Line Of Credit, dataset consist of borrower's past transaction, inquires and other states and their success/failure of lending loans.
+ `ExternalRiskEstimate` - consolidated indicator of risk markers (equivalent of polish BIK’s rate)
+ `MSinceOldestTradeOpen` - number of months that have elapsed since first trade
+ `MSinceMostRecentTradeOpen` - number of months that have elapsed since last opened trade
+ `AverageMInFile` - average months in file
+ `NumSatisfactoryTrades` - number of satisfactory trades
+ `NumTrades60Ever2DerogPubRec` - number of trades which are more than 60 past due
+ `NumTrades90Ever2DerogPubRec` - number of trades which are more than 90 past due
+ `PercentTradesNeverDelq` - percent of trades, that were not delinquent
+ `MSinceMostRecentDelq` - number of months that have elapsed since last delinquent trade
+ `MaxDelq2PublicRecLast12M` - the longest delinquency period in last 12 months
+ `MaxDelqEver` - the longest delinquency period
+ `NumTotalTrades` - total number of trades
+ `NumTradesOpeninLast12M` - number of trades opened in last 12 months
+ `PercentInstallTrades` - percent of installments trades
+ `MSinceMostRecentInqexcl7days` - months since last inquiry (excluding last 7 days)
+ `NumInqLast6M` - number of inquiries in last 6 months
+ `NumInqLast6Mexcl7days` - number of inquiries in last 6 months (excluding last 7 days)
+ `NetFractionRevolvingBurden` - revolving balance divided by credit limit
+ `NetFractionInstallBurden` - installment balance divided by original loan amount
+ `NumRevolvingTradesWBalance` - number of revolving trades with balance
+ `NumInstallTradesWBalance` - number of installment trades with balance
+ `NumBank2NatlTradesWHighUtilization` - number of trades with high utilization ratio (credit utilization ratio - the amount of a credit card balance compared to the credit limit)
+ `PercentTradesWBalance` - percent of trades with balance


### Task & Goal
There are 2 aims for this.
1. Improve performance.
   + Proposal baseline has 70.77% in F1, 73% in Accuracy

2. Find features which best explains the model

## Proposed Methods

### 1. Improving Performance
1. Tree-based Classifiers
   + Find the best classifier with TPOT
   + Other AutoML libraries to reduce labouring cost

2. Neural Networks
   
   I also tried deep neural networks, since the proposal insisted that deep learning models does not have enoguth interpretability nor satisfying performance from them. However, from [this survey](https://arxiv.org/pdf/2110.01889.pdf) one point out that there are numerous deep learning architectures proposed to deal with tabular data, even with explainability. Models below are the models that are widely used in the tabular data. I have only used **TabNet** in this project.
   + [**TabNet**](https://arxiv.org/pdf/1908.07442.pdf) [Code Implementation](https://github.com/dreamquark-ai/tabnet)
   + [**Tab Transformer**](https://arxiv.org/pdf/2012.06678.pdf) [Code Implementation](https://github.com/lucidrains/tab-transformer-pytorch)
   + [**SAINT**](https://arxiv.org/pdf/2106.01342.pdf) [Code Implementation](https://github.com/somepago/saint)

    Another reason for using deep neural networks is the feature. No features are directly related to the target label from the view. From the correlation map, one can also observe that there are many redundant features with each other. Take a look at the pairplot and correlation heatmap below (click to enlarge).
    ![image](./assets/pairplot.png)
    ![image](./assets/correlation_heatmap.png)
    For this reason, I here suggest that **non-linearity should come to play**.

### 2. Finding features
From [Yu Zhang](https://arxiv.org/pdf/2012.14261.pdf), interpretability of the neural networks are as follows.
![image](./assets/image1.png)


Since this work tries to find a general trends in predicting the successful lending, I have focused on using passive and global method for inspecting the features. Since LIME and Shap are suitable for finding such interpretations, I have used these libraries to find the features that best explains this.

Furthermore, I add i.i.d. Gaussian noise to features one by one, which is example of a perturbation method, and find the feature that disrupts performance the most.

## Results
### Tree-based Classifiers
Through TPOT, we got the best model of XGBoost with on my train/dev split with 68% F1-score on dev. Tree-based classifiers support feature importance so therefore I plot the best models' feature importance below.
![image](./assets/feature_importance_xgboost.png)

Also I have arranged 10 fold in training set and test with hold-out dev set, then retrieved feature importances of the folds. We can use mean and variance of these results and its plot is shown below.
![image](assets/feature_importance_10fold_xgboost.png)

`ExternalRiskEstimate` shows the most importance in both results. `MSinceMostRecentInqexcl7days` and `NetFractionRevolvingBurden` are followed by the models regardness of the most import feature. Other features are competing each other with high variance, or doesn't show much importance compared to the first listed 3 features.

### TabNet

In **TabNet**, one can give variations of parameters in number of dimensions, attentions and steps. Here I have manually searched the best parameters with F1-score and accuracy. The results are shown belown.
![image](./assets/f1_tabnet_wo_pp.png)
![image](./assets/acc_tabnet_wo_pp.png)

I have used the best F1-score which configures as `n_d=n_a=32`, `n_steps=7`. TabNet supports feature importance and explanation matrix. Former represents the models' feature importance in global point of view - by regarding the whole dataset, while the latter contains (`# instances`, `# features`) matrix of importance, i.e. contains all the feature importance by instance.

The first image below shows the feature importance of the TabNet.
![image](./assets/feature_importance_tabnet.png)

The second image below shows the explanation matrix, through calculating the whole mean and variance among all instances. We can see much variance here.
![image](assets/mean_explanation_matrix_tabnet.png)

Both the feature importance and explanation matrix shows consistent results in that they have 3 most importance features - `ExternalRiskEstimate`, `MSinceMostRecentInqexcl7days` and `NetFractionRevolvingBurden`. Other features are almost agrees in the order except that few features are competing.

## Conclusion

In this work, I have tried to improve the performance by 2 different family of models - tree-based and MLP. From hold-out dev data result, MLP-based model, TabNet, outperformed tree-based model with 71.8% over 68%. With these models, I have found out that the most important feature was `ExternalRiskEstimate`, which resides with the previous works of FICO competition in interpretability. Other important features in predicting the successful loaning was `MSinceMostRecentInqexcl7days` and `NetFractionRevolvingBurden` and these results are consistent in both tree-based and TabNet result as well. Since most of the features are not hand-crafted in detail, there must be a model that better performs than 71.8% in F1-score.

## References

HELOC
+ [XAI Stories](https://pbiecek.github.io/xai_stories/story-heloc-credits.html)
+ [FICO Community Blog](https://community.fico.com/s/blog-post/a5Q2E0000001czyUAA/fico1670)
+ [An Interpretable Model with Globally Consistent Explanations for Credit Risk](https://users.cs.duke.edu/~cynthia/docs/ChenEtAlFICO2018.pdf)
+ [Explainable AI for Interpretable Credit Scoring](https://arxiv.org/ftp/arxiv/papers/2012/2012.03749.pdf)

Models
+ [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/pdf/1908.07442.pdf)
+ [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/1908.07442.pdf)
+ [Deep Neural Networks and Tabular Data: A Survey](https://arxiv.org/pdf/2110.01889.pdf)
  
Interpretability
+ [A Survey on Neural Network Interpretability](https://arxiv.org/pdf/2012.14261.pdf)