# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:00:35 2021

@author: iseabrook1
"""
#This script contains the functions required to assess the predictability of  
#subsequent node presence from the node level metrics m_{a-c},
# eigenvector centrality, pagerank, degree and community label.

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Community aware evaluation of node importance
#
################################################################################
#   Instructions for use.
#   The function node_change_prediction is a wrapper to produce the results shown 
#   in Seabrook et. al.. This function first of all splits the data into train, 
#   test and validation sets, before making use of setup_preprocessor to initiate 
#   the predictor preprocessing steps. It then allows the user to compare the 
#   performance of two models - one random forest classifier and one logistic 
#   regression classifier, both for which the parameter values are selected using 
#   5-fold cross validation. 
#   Following selection of the best classifier, the function then evaluates the 
#   model performance on the test set, printing the results, and then calculates 
#   the permutation feature importance for each of the node level metrics, outputting
#   a bar plot to allow visual comparison of the feature importances. 

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.inspection import permutation_importance

def setup_preprocessor(cont_cols, cat_cols, ord_cols, bin_cols):
    """ Function to build preprocessor transformer, to form the first step
    of the model pipeline. 
    Parameters:
        cont_cols (list(str)): list of strings for continuous column names
        cat_cols (list(str)): list of strings for continuous column names
        ord_cols (list(str)): list of strings for continuous column names
        bin_cols (list(str)): list of strings for continuous column names
    Returns:
        ColumnTranformer object: preprocessor
    """
    numerical_transformer = Pipeline(steps = [('imputer',\
                                               SimpleImputer(missing_values = np.nan,\
                                                             strategy = 'most_frequent')),
                                                ('scaler',\
                                                 StandardScaler())])                                        
    categorical_transformer = Pipeline(steps = [
                                                ('onehot',\
                                                 OneHotEncoder(handle_unknown = 'ignore'))
                                               ])
    #mapping the ordinals to a -1, 1 range will help with speed of learning weights                                          
    ordinal_transformer = Pipeline(steps = [('imputer',\
                                             SimpleImputer(missing_values = np.nan,\
                                                           strategy='most_frequent')),
                                                ('scaler', MinMaxScaler(feature_range=(0,1)))
                                            ])
    binary_transformer = Pipeline(steps=[('imputer',\
                                          SimpleImputer(missing_values = np.nan,\
                                                        strategy='constant',\
                                                            fill_value=0))])
    preprocessor = ColumnTransformer(transformers = [('num',\
                                                      numerical_transformer,\
                                                          cont_cols),
                                                    ('cat',\
                                                     categorical_transformer,\
                                                         cat_cols), 
                                                     ('ord',\
                                                      ordinal_transformer,\
                                                          ord_cols),
                                                     ('bins',\
                                                      binary_transformer,\
                                                          bin_cols)
                                                    ])
    return(preprocessor)
    
def pipeline_model(model,classifiers_dict, classifier_params_dict, preprocessor, X_train, y_train, X_test, y_test, scoring='balanced_accuracy'):
    """ Function to build pipeline for a given classification model and processor,
    to run 5 fold cross validation to select the best parameters according to 
    balanced accuracy score.
    Parameters:
        model: sklearn classifier, with 'fit' and 'transform' methods
        classifiers_dict: dict containing the possible models to consider
        classifiers_params_dict: dict containing the possible parameters for
        the models
        preprocessor: preprocessor object produced using setup_preprocessor
    Returns:
        prints of best parameters for the model, and the balanced accuracy score
        for the parameter selection.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('sampler', RandomOverSampler()),        

        ('clf', classifiers_dict[model]),
    ])
    grid_params_lr = classifier_params_dict[model]
    grid = GridSearchCV(pipeline, grid_params_lr, cv =5, \
                        scoring = scoring, n_jobs=-1) 
    grid.fit(X_train, y_train) 
    print("best parameters: ", grid.best_params_) 
    grid.refit 
    print(scoring+':', grid.best_score_) 
    return(grid.best_params_, grid.best_score_)

def train_test_split_sorted(X, y, test_size, dates):
    """Splits X and y into train and test sets, with test set separated by most recent dates.

    Example:
    --------
    >>> from sklearn import datasets

    # Fake dataset:
    >>> gen_data = datasets.make_classification(n_samples=10000, n_features=5)
    >>> dates = np.array(pd.date_range('2016-01-01', periods=10000, freq='5min'))
    >>> np.random.shuffle(dates)
    >>> df = pd.DataFrame(gen_data[0])
    >>> df['date'] = dates
    >>> df['target'] = gen_data[1]

    # Separate:
    >>> X_train, X_test, y_train, y_test = train_test_split_sorted(df.drop('target', axis=1), df['target'], 0.33, df['date'])

    >>> print('Length train set: {}'.format(len(y_train)))
    Length train set: 8000
    >>> print('Length test set: {}'.format(len(y_test)))
    Length test set: 2000
    >>> print('Last date in train set: {}'.format(X_train['date'].max()))
    Last date in train set: 2016-01-28 18:35:00
    >>> print('First date in test set: {}'.format(X_test['date'].min()))
    First date in test set: 2016-01-28 18:40:00
    """

    n_test = ceil(test_size * len(X))

    sorted_index = [x for _, x in sorted(zip(np.array(dates), np.arange(0, len(dates))), key=lambda pair: pair[0])]
    train_idx = sorted_index[:-n_test]
    test_idx = sorted_index[-n_test:]

    if isinstance(X, (pd.Series, pd.DataFrame)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]

    return X_train, X_test, y_train, y_test

def node_change_prediction(dataset, X,y, date):
    ''' Function to act as a wrapper to select, train and evaluate a classifier which predicts node 
    level changes. Evaluation is done according to comparison of precision and 
    recall to the same metrics for a dummy classifier which randomly predicts changes
    in proportion to the dataset prior. This function makes use of setup_preprocessor
    and pipeline_model.
    Parameters:
        dataset: pandas dataframe with columns containing our four node importance
        metrics, eigenvector centrality, pagerank, degree and centrality for each node.
        X: subset of dataset containing uncorrelated features to pass  to the classifier
        y: boolean label for node changes - target variable for the classifier. 
    Returns:
        function prints out precision and recall for the classifier, precision,
        recall and the associated error for the dummy classifier, plot of the 
        permutation feature importances.
    '''
    cat_cols = []
    ord_cols = [] 
    cont_cols = X.columns
    bin_cols = []
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,\
#                                                         random_state=42,\
#                                                             stratify  = y)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,\
#                                                       test_size=0.25,\
#                                                           random_state=42,\
#                                                               stratify = y_train)
    X_train, X_test, y_train, y_test = train_test_split_sorted(X, y, test_size=0.2, dates = date)
    X_train, X_val, y_train, y_val = train_test_split_sorted(X_train, y_train,\
                                                      test_size=0.25, dates = date[:len(X_train)])

    ############### Step 2 - set up preprocessor and pipeline ################
    preprocessor = setup_preprocessor(cont_cols, cat_cols, ord_cols, bin_cols)
    classifiers_dict = {'rf':RandomForestClassifier(),
                    'lr':LogisticRegression(class_weight='balanced',\
                                            random_state = 42)} 
    classifier_params_dict={'rf':
                        {'clf__bootstrap': [False, True],
                        'clf__n_estimators': [80,100, 130]},
                  'lr':
                        {'clf__C': [0.001,0.01,0.1,1,10, 100],
                        'clf__penalty': (['l2']),
                        'clf__max_iter': [80, 100, 200]}
                       }
    ####### Step 3 - 5 fold CV for parameter selection for chzosen models ######
    models=['rf', 'lr']
    for i in models:
        best_params, m = pipeline_model(i, classifiers_dict, classifier_params_dict, preprocessor, X_train, y_train, X_val, y_val)
    print("input 0 if the random forest produced the best, 1 if the logistic regression did.")
    classifier_indicator = input()
    print(best_params)
    if classifier_indicator == 0:
        kn_classifier = RandomForestClassifier(bootstrap = best_params.get('clf__bootstrap'), n_estimators = best_params.get('clf__n_estimators'))
    else:
        kn_classifier = LogisticRegression(C= best_params.get('clf__C'), max_iter= best_params.get('clf__max_iter'), penalty=best_params.get('clf__penalty'),  n_jobs=-1)
    pipeline = Pipeline(steps =  [('preprocessor', preprocessor), 
                              ('sampler', RandomOverSampler()),        
                              ('kn_classifier', kn_classifier)
                             ]) 
    pipeline.fit(X_train, y_train)
    y_prediction = pipeline.predict(X_test)   
    recall = recall_score(y_test, y_prediction)
    print("unshuffled dataset")
    prec = precision_score(y_test, y_prediction)
    print("Precision:", prec)    
    print("Recall : ", recall)    
    tn, fp, fn, tp = confusion_matrix(y_test, y_prediction).ravel()   
    print("tn:",tn, "fp:", fp,"fn:", fn,"tp:",tp) 
    #binomial monte carlo generation, attempt to use log reg to predict. 
    recall_dumm_list=[]
    precision_dumm_list=[]   
    for j in range(100):
        ds_le_dumm = dataset.copy()
        p = len(ds_le_dumm[ds_le_dumm.change_bool==1])/len(ds_le_dumm)
        ds_le_dumm.loc[:, "change_bool"] = np.random.binomial(1, p, len(ds_le_dumm))
        X_dumm = ds_le_dumm[cont_cols].fillna(0)
        y_dumm = ds_le_dumm["change_bool"]
        X_train_dumm, X_test_dumm, y_train_dumm, y_test_dumm = train_test_split(X_dumm, y_dumm, test_size=0.2, random_state=42, stratify  = y)
        pipeline_dumm = Pipeline(steps =  [ ('preprocessor', preprocessor),
                                        ('sampler', RandomOverSampler()),        
    
                                        ('gb_classifier', kn_classifier)
                                        ]) 
        pipeline_dumm.fit(X_train_dumm, y_train_dumm)
        y_prediction_dumm = pipeline_dumm.predict(X_test_dumm) 
        precision_dumm = precision_score(y_test_dumm, y_prediction_dumm)
        recall_dumm = recall_score(y_test_dumm, y_prediction_dumm)
        precision_dumm_list.append(precision_dumm)
        recall_dumm_list.append(recall_dumm)
        recall_dumm_list.append(recall_dumm)
    recall_mean = np.mean(recall_dumm_list)
    precision_mean = np.mean(precision_dumm_list)
    recall_std = np.std(recall_dumm_list)
    precision_std = np.std(precision_dumm_list)
    print('Precision dumm: %.3f' % precision_mean)
    print('Recall dumm: %.3f' % recall_mean)  
    prec_err = 1.96*precision_std/10
    recall_err = 1.96*recall_std/10
    print("PRAD", prec_err)
    print("ROCAD", recall_err)
    ###feature importance
    pipeline.named_steps["preprocessor"].transform(X_train)
    X_train_transformed = pipeline.named_steps["preprocessor"].transform(X_train)
    X_train_transformed=pd.DataFrame(X_train_transformed, index=X_train.index, columns =cont_cols)
    X_train_transformed, y_train_transformed = pipeline.named_steps["sampler"].fit_sample(X_train_transformed, y_train)
    imps = permutation_importance(LogisticRegression(C= 0.001, max_iter= 80, penalty='l2').fit(X_train_transformed, y_train_transformed),X_train_transformed, y_train_transformed)
    importance = imps.importances_mean
    sorted_idx = importance.argsort()
    y_ticks = np.arange(0, len(cont_cols))
    fig, ax = plt.subplots(figsize = (12,10))
    ax.barh(y_ticks, importance[sorted_idx], xerr=imps.importances_std[sorted_idx],capsize=20)
    ax.set_yticklabels(np.array(cont_cols)[sorted_idx])
    plt.grid()
    ax.set_yticks(y_ticks)
    fig.tight_layout()
    plt.show() 
