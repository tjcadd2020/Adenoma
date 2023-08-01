import pandas as pd
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import GroupKFold,StratifiedShuffleSplit, StratifiedKFold, train_test_split,KFold,cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import plot_roc_curve,roc_curve,auc,accuracy_score, recall_score, precision_score, average_precision_score, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from sklearn.inspection import permutation_importance
from scipy import interp
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import copy
import cloudpickle as pickle
import matplotlib.pyplot as plt

def get_kfold_auc_shuffle_rf(data, meta_group):
    aucs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    plot_data = []
    i = 0
    splitor = StratifiedKFold(n_splits=5, shuffle=True,random_state=SEED) 
    clf = RandomForestClassifier(n_estimators = 101, oob_score = True, random_state =SEED,
                                max_features = 0.25,class_weight = 'balanced')
    
    for train_index, test_index in splitor.split(data, meta_group):
        y_train, y_test = meta_group[train_index], meta_group[test_index]
        X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
        
        #ros = RandomOverSampler(random_state = SEED)
        #X_train, y_train = ros.fit_resample(X_train0, y_train0)
        
        probas = clf.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ### plot data
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plot_data.append([fpr, tpr, 'ROC Fold %d(AUC = %0.2f)' %(i+1, roc_auc)])
        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_auc, clf, (plot_data, mean_fpr, mean_tpr, tprs, np.std(aucs))

def feature_auc_shuffle_rf(data):
    featureauc = pd.DataFrame(columns = ['auc'],index = list(data.columns))
    for j in list(data.columns):
        data_species = pd.DataFrame(data.loc[:,j])
        featureauc.loc[j,:] = get_kfold_auc_shuffle_rf(data_species,y_data)[0]
    featureauc = featureauc.sort_values('auc',ascending = False)
    return featureauc

def get_kfold_auc_op(data, meta_group, **params):
    aucs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    plot_data = []
    i = 0
    splitor = StratifiedKFold(n_splits=5, shuffle=True,random_state=SEED) 
    clf = RandomForestClassifier(random_state=SEED,class_weight = 'balanced').set_params(**params)
    
    for train_index, test_index in splitor.split(data, meta_group):
        y_train, y_test = meta_group[train_index], meta_group[test_index]
        X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
        
        probas = clf.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ### plot data
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plot_data.append([fpr, tpr, 'ROC Fold %d(AUC = %0.2f)' %(i+1, roc_auc)])
        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_auc

def bayesian_optimise_rf(X, y, clf_kfold, n_iter, init_points = 5):
    def rf_crossval(n_estimators, max_features,max_depth,max_samples):
        return clf_kfold(
            data = X,
            meta_group = y,
            n_estimators = int(n_estimators),
            max_samples = max(min(max_samples,0.999),1e-3),
            max_features = max(min(max_features, 0.999), 1e-3),
            max_depth = int(max_depth),
            bootstrap = True
        )
    
    optimizer = BayesianOptimization(
        random_state = SEED,
        f = rf_crossval,
        pbounds = {
            "n_estimators" : (10, 500),
            "max_features" : (0.1, 0.999),
            "max_samples" : (0.1,0.999),
            "max_depth" : (1,5)
        }
    )
    optimizer.maximize(n_iter = n_iter , init_points = init_points)
    print("Final result:", optimizer.max)
    return optimizer.max

def feature_imps(param,data,y_data):
    rf = RandomForestClassifier(random_state=SEED).set_params(**param).fit(data.values,y_data)
    result = permutation_importance(rf, data.values, y_data,
                                    n_repeats = 10, random_state = SEED)
    perm_sorted_idx = result.importances_mean.argsort()
    
    tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
    tree_indices = np.arange(0,len(rf.feature_importances_)) + 0.5
    fig,(ax1,ax2) = plt.subplots(1,2, figsize = (18,20))
    ax1.barh(tree_indices, rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(data.columns[tree_importance_sorted_idx])
    ax1.set_ylim((0, len(rf.feature_importances_)))
    ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                labels=data.columns[perm_sorted_idx])
    fig.tight_layout()
    return result,tree_importance_sorted_idx,tree_indices, perm_sorted_idx,plt

#feature selection

featureauc_data = feature_auc_shuffle_rf(data)
featureauc_tmp = featureauc_data.loc[featureauc_data['auc'] > 0.5]
data = data.loc[:,featureauc_tmp.index]

corre_data = feature_corre(data,data.T,0.7)
data = data.loc[:,corre_data.index]

select = list(data.columns)
best_auc = 0
best_plot_data = []
best_features = []
feature_rank = []
while(len(select)>1):
    aucs = []
    for ni in select:
        temp = copy.deepcopy(select)
        temp.remove(ni)
        roc_auc, _, plot_data = get_kfold_auc_shuffle_rf(data.loc[:, temp], y_data)
        aucs.append([temp, roc_auc, plot_data])
        #print(temp, roc_auc)
    select, roc_auc, plot_data = sorted(aucs, key=lambda x:x[1], reverse = True)[0]
    if roc_auc >= best_auc:
        best_auc = roc_auc
        best_features = select
        best_plot_data = plot_data
    feature_rank.append([select, roc_auc])
    print('### Best AUC :', len(select), round(best_auc, 3), round(roc_auc, 3))

data= data.loc[:,best_features]
print(len(best_features),data.shape,len(y_data))


#hyperparameter adjusting
tune_result=bayesian_optimise_rf(data,y_data,get_kfold_auc_op,100)
tune_result['params']['n_estimators'] = int(tune_result['params']['n_estimators'])
best_param= tune_result['params']

model_result = get_kfold_auc(data.values, y_data, best_param, k=5)
print("auc:",model_result[0],"sen:",model_result[-2],"spe:",model_result[-3],"mcc",model_result[-1])

#feature importance
result,tree_importance_sorted_idx,tree_indices, perm_sorted_idx,plt = feature_imps(best_param,data,y_data)

