import seaborn as sns
from matplotlib import pyplot
import scikitplot as skplt
from sklearn import metrics
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import shap
import logging
import matplotlib
import commenFuncs
from numpy import std
import pickle as pk
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost
from sklearn.metrics import mean_absolute_error
matplotlib.use('Agg')
logger = logging.getLogger()
import time

def lgmTrain(data, params, outFileName, trainSize,features):
    import lightgbm as lgb

    X_train, X_train_Wei, Y_train, X_test, X_test_Wei, Y_test, X_vali, X_vali_Wei, Y_vali = commenFuncs.prepareTrainTestVali(
        data, trainSize)
    lgb_train = lgb.Dataset(X_train, label=Y_train, weight=X_train_Wei)
    print("dataset  ", lgb_train)
    lgb_vali = lgb.Dataset(X_vali, label=Y_vali,
                           weight=X_vali_Wei, reference=lgb_train)

#**********************#
    models = get_models()
# evaluate the models and store results
    results, names = list(), list()
   # for name, model in models.items():
   #     scores = evaluate_model(model,X_train, Y_train)
   #     results.append(scores)
   #     names.append(name)
   #     print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    # plot model performance for comparison
   # pyplot.boxplot(results, labels=names, showmeans=True)
   # pyplot.show()
#############################################
    logger.info("Start training...")
    evals_result = {}
    print("params: ", params)
    gbm = lgb.train(params, lgb_train, valid_sets=lgb_vali,
                    evals_result=evals_result)

    logger.info("Save model...")
    gbm.save_model(outFileName + '/lightgbm.model')

    # Plot the metrics
    logger.info('Plotting metrics recorded during training...')
    metric_name = params['metric']
    lgb.plot_metric(evals_result, metric=metric_name)
    plt.savefig(outFileName + '/lightgbm_metrics.png')
    plt.close()

    # Plot shap
    shap_values = shap.TreeExplainer(gbm).shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    plt.savefig(outFileName + '/lightgbm_shap.png')

    #Plot the importance
    logger.info('Plotting the variable importance...')
    plt.figure(figsize=(10, 10))
    lgb.plot_importance(gbm, max_num_features=20)
    plt.title("Feature Importances")
    plt.savefig(outFileName + '/lightgbm_importance.png')
    plt.close()
    #plot importance 2
    print(gbm.feature_importance(), features)
    feature_importance_df = pd.DataFrame()
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = gbm.feature_importance()
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False).index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',ascending=False)
    plt.figure(figsize=(10,10))
    sns.barplot(y="Feature",
            x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outFileName+ '/lgb_importances.png')
    
    # Plot the tree structure
  #  graph = lgb.create_tree_digraph(gbm, tree_index=3, name='tree structure')
  #  graph.render(outFileName + '/lightgbm_tree')

    # Prediction, envaluate the model and plot ROC
    logger.info("Start prediction")
    Y_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    Y_prob_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_pred = []
    if params['num_class'] == 1:  # binary
        for x in Y_prob:
            if x > 0.5:
                Y_pred.append(1)
            else:
                Y_pred.append(0)
        fpr, tpr, thread = metrics.roc_curve(Y_test, Y_prob)
        fpr_train, tpr_train, thread_train = metrics.roc_curve(
            Y_train, Y_prob_train)
        roc_lightgbm = []
        roc_lightgbm.append(fpr_train)
        roc_lightgbm.append(tpr_train)
        roc_lightgbm.append(np.array(metrics.roc_auc_score(Y_train,Y_prob_train)))
        roc_lightgbm.append(fpr)
        roc_lightgbm.append(tpr)
        roc_lightgbm.append(np.array(metrics.roc_auc_score(Y_test,Y_prob)))
        roc_lightgbm = np.array(roc_lightgbm)
        np.save('result/roc_lightgbm.npy',roc_lightgbm)
        plt.figure(figsize=(9, 9))
        plt.plot(fpr, tpr, label='Test ROC curve (area = %0.2f)' %
                 metrics.roc_auc_score(Y_test, Y_prob))
        plt.plot(
            fpr_train, tpr_train, label='Train ROC curve (area = %0.2f)' % metrics.roc_auc_score(Y_train, Y_prob_train))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
    else:  # multi-class
        #Y_pred = [list(x).index(max(x)) for x in Y_prob]
        skplt.metrics.plot_roc(Y_test, Y_prob)
        print("*************************************multi-class**************************************")

#    logger.info("AUC score: {:<8.5f}".format(metrics.accuracy_score(Y_test, Y_pred)))
#    print("accuracy", metrics.accuracy_score(Y_test, Y_pred))
#    print("Mean Absolute Error : " + str(mean_absolute_error(Y_test, Y_pred)))
    # Plot the ROC curve
    plt.savefig(outFileName + '/lightgbm_ROC.png')
    plt.close()
#-----------------------------#
# Confusion matrix 
#-----------------------------#
#    skplt.metrics.plot_confusion_matrix(Y_test,Y_prob,figsize=(8,6)); plt.show()
#    plt.savefig(outFileName + '/lightgbm_confuse.png')
#------------------------------#
#correlation matrix
#------------------------------#
    print()
    print('Correlation Matrix of All Numerical Features')
    fig = pyplot.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(data.corr(),
                     vmin=-1, vmax=1,
                     interpolation='none')
    fig.colorbar(cax)
    ticks = np.arange(0, 24, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    pyplot.title("correlation matrix (square)")
    pyplot.savefig(outFileName+'/matrix_complete.png')
    pyplot.close()
#ax.set_xticklabels(colNumeric)
#ax.set_yticklabels(colNumeric)

# ---------------------------------------
# Correlation Plot using seaborn
# ---------------------------------------

    print()
    print("Correlation plot of Numerical features")

# Compute the correlation matrix

    corr = data.corr()
    print("corr", corr)

# Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

# Set up the pyplot figure
    f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, annot_kws={"size": 7}, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0,
                center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .5})
    pyplot.title("correlation matrix (triangle)")
    pyplot.savefig(outFileName+'/matrix_triangle.png')
    pyplot.close()
#
# USING histograms
#    j = 1
#    print()
#    print('Histogram of each Numerical Feature')
#    pyplot.figure(figsize=(11,9))
#    for col in colNumeric:
#        plt.subplot(3,3,j)
#        plt.axis('on')
#        plt.tick_params(axis='both', left='on', top='off',
#            right='off', bottom='on', labelleft='off', labeltop='off',
#            labelright='off', labelbottom='off')
#
#        dataset[col].hist()
#        j += 1
#    pyplot.title("histograms")
#    pyplot.savefig(outFileName+'/histograms.png')
#    pyplot.close()


def xgbTrain(data, params, outFileName, trainSize,features):
    import xgboost as xgb
    start_time = time.time()
    X_train, X_train_Wei, Y_train, X_test, X_test_Wei, Y_test, X_vali, X_vali_Wei, Y_vali = commenFuncs.prepareTrainTestVali(
        data, trainSize)

    xgb_train = xgb.DMatrix(X_train, label=Y_train, weight=X_train_Wei)
    xgb_vali = xgb.DMatrix(X_vali, label=Y_vali, weight=X_vali_Wei)
    xgb_test = xgb.DMatrix(X_test, label=Y_test, weight=X_test_Wei)
    ############################################################
    #model = xgboost.XGBClassifier(objective="binary:logistic")
    model = xgboost.XGBClassifier(objective="multi:softprob")
    parameters = {
        'max_depth':[6],# [4,5,6],
        'gamma':[1], # [0.5,1,2],
        'learning_rate':[0.05], # [0.05, 0.01],
        'n_estimators':[50] # [30,50,70]
    }

    grid = GridSearchCV(estimator=model, param_grid=parameters, verbose=1, n_jobs=-1, cv = 6, refit=True)
    grid.fit(X_train,Y_train)
    # Results from Grid Search
    print("\n========================================================")
    print(" Results from Grid Search " )
    print("========================================================")
    print("\n The best estimator across ALL searched params:\n",
          grid.best_estimator_)
    print("\n The best score across ALL searched params:\n",
          grid.best_score_)
    print("\n The best parameters across ALL searched params:\n",
          grid.best_params_)
    print("\n ========================================================")
    model = grid.best_estimator_
    #**** validation *****#
    cv_results = cross_val_score(model, X_train, Y_train, cv = 6, scoring = 'accuracy',
                                 n_jobs = -1, verbose = 1)
    print()
    print("Cross Validation results: ", cv_results)
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (cv_results.mean(), cv_results.std())
    print(prt_string)
    
    # Final fitting of the Model
    model.fit(X_train, Y_train)

    print(); print('========================================================')
    print(); print(model.get_params(deep = True))
    print(); print('========================================================')
    
    #** evaluate model **#
    pred_Class          = model.predict(X_test)
    acc                 = accuracy_score(Y_test, pred_Class)
    classReport         = classification_report(Y_test, pred_Class)
    confMatrix          = confusion_matrix(Y_test, pred_Class)
    kappa_score         = cohen_kappa_score(Y_test, pred_Class)

    print(); print('Evaluation of the trained model: ')
    print(); print('Accuracy : ', acc)
    print(); print('Kappa Score : ', kappa_score)
    print(); print('Confusion Matrix :\n', confMatrix)
    print(); print('Classification Report :\n',classReport)

    prod_Ytrain = model.predict_proba(X_train)[:,1]
    prod_Ytest = model.predict_proba(X_test)[:,1]
    pred_proba = model.predict_proba(X_test)
    print("*** Y_train***", prod_Ytrain)
    # Add more plots here using scikit-plot
    
    # ROC curves
    if False:
        fpr, tpr, thread = metrics.roc_curve(Y_test, prod_Ytest)
        fpr_train, tpr_train, thread_train = metrics.roc_curve(Y_train, prod_Ytrain)
        roc_xgboost = []
        roc_xgboost.append(fpr_train)
        roc_xgboost.append(tpr_train)
        roc_xgboost.append(np.array(metrics.roc_auc_score(Y_train,prod_Ytrain)))
        roc_xgboost.append(fpr)
        roc_xgboost.append(tpr)
        roc_xgboost.append(np.array(metrics.roc_auc_score(Y_test,prod_Ytest)))
        roc_xgboost = np.array(roc_xgboost)
        np.save('result/roc_xgboost.npy',roc_xgboost)
        plt.figure(figsize=(9, 9))
        plt.plot(fpr, tpr, label='Test ROC curve (area = %0.2f)' %
                 metrics.roc_auc_score(Y_test, prod_Ytest))
        plt.plot(
            fpr_train, tpr_train, label='Train ROC curve (area = %0.2f)' % metrics.roc_auc_score(Y_train, prod_Ytrain))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        # skplt.metrics.plot_roc(Y_test,pred_proba,figsize=(8,6)); plt.show()
    else:  # multi-class
        #Y_pred = [list(x).index(max(x)) for x in prod_Ytest]
        skplt.metrics.plot_roc(Y_test,pred_proba)
    plt.savefig(outFileName + '/xgboost_roc.png')
    
    # Confusion matrix
    skplt.metrics.plot_confusion_matrix(Y_test,pred_Class,figsize=(8,6)); plt.show()
    plt.savefig(outFileName + '/xgboost_confuse.png')
    
    #importance
    print(model.feature_importances_)
    xgb.plot_importance(model)
    #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.savefig(outFileName + '/xgboost_importance.png')
    
    # precision recall curve
    skplt.metrics.plot_precision_recall(Y_test, pred_proba,
            title='Precision-Recall Curve', plot_micro=True,
            classes_to_plot=None, ax=None, figsize=(8,6),
            cmap='nipy_spectral', title_fontsize='large',
            text_fontsize='medium'); plt.show()
    plt.savefig(outFileName + '/xgboost_metrics.png')
    
# correlation matrix
    print()
    print('Correlation Matrix of All Numerical Features')
    fig = pyplot.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(data.corr(),
                     vmin=-1, vmax=1,
                     interpolation='none')
    fig.colorbar(cax)
    ticks = np.arange(0, 24, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
# ---------------------------------------
# Correlation Plot using seaborn
# ---------------------------------------

    print()
    print("Correlation plot of Numerical features")

# Compute the correlation matrix

    corr = data.corr()
    print("corr", corr)

# Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

# Set up the pyplot figure
    f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, annot_kws={"size": 7}, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0,
                center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .5})
    pyplot.title("correlation matrix (triangle)")
    pyplot.savefig(outFileName+'/xgboost_triangle.png')
    pyplot.close()
    #****save model ****#
    with open(outFileName+'/xgboost_model.pickle', 'wb') as f:
            pk.dump(model, f)
    #**** time ******#
    print()
    print("Required Time %s seconds: " % (time.time() - start_time))
                                  

#####################################

# Use the most complicate RNN model 'LSTM' because it's the most accurate one.
# Alternative choices: 'GRU'(No forget gate. ~2/3 simpler with only small accuracy lost) and 'simpleRNN'(most simple one, no gate)
def rnnTrain(data, params, outFileName, trainSize):
	import tensorflow as tf
	import tensorflow.keras as ks


def get_models():
	models = dict()
	trees = [10, 50, 100, 500, 1000, 5000]
	for n in trees:
		models[str(n)] = LGBMClassifier(n_estimators=n)
	return models
# evaluate a give model using cross-validation


def evaluate_model(model, X_train, Y_train):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train, Y_train,
	                         scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

