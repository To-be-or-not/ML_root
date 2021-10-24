# setup.sh
  setup the environment for BDT training, install some packages required for the framework

# config.json
This file is to set paramaters for BDT training
paramaters:
    train: 
        ==> ofter BDT training trees, if we want multiclass, we can set more elements. Generally, it looks like:
            [[BkgTree11,BkgTree12...],[BkgTree21,BkgTree22]...[SigTree11,SigTree22]]
    prediction:
        ==> prediction:
            ofter trees to be predicted.
    out_branch:
        ==> out_branch: 
            set out_branch names
    variables:
        ==> training variables 
    totweight, lumiScaling:
        ==> variabls to calculate weights, but I just set them in commenFuncs.py.
    largestAllowWei:
        ==> set a upper limit for event's weight 
    lightgbm:
        ==> paramaters for lightgbm training (you can add more and change them as you like)
            ==> objective: binary/multiclass
            ==> metric: binary_logloss/multi_logloss
            ==> others: to control the size of the model
    XGboost:
        ==> paramaters for XGboost training (you can add more and change them as you like)
            ==> objective: binary/multiclass
            ==> metric: binary_logloss/multi_logloss
            ==> others: to control the size of the model


## analysis.py
This file is to control training and prediction:
    ==> training + prediction
    ==> just training to get a good model
    ==> just prediction using a model we have had
NOTE:
    ==> XGboost do not support negative weight ==> we can set noNegWei = True 
         when ifweight=False ==> we just set weight=1

#training.py 
This file defines lightTrain and xgbTrain training framework
    lightgbm:
        ==> train a model and save it
        ==> plot some performance plot:
            ==> roc, importance, metric, correlation(Please pay attention to the plot size if you find they are inporper)
    xgboost:
        ==>Here, so far, we did not use the training paramaters "xgboost". We are using GridSearchCV to scan paramaters vector to get a good combination.
        ==> paramaters grid is large, this training will cost much time. 
        ==> plot some performance plot:
            ==> roc, importance, metric, correlation(Please pay attention to the plot size if you find they are inporper) 
NOTE:
    will tidy up this part and add neutral networks training method 

# prediction.py
This file is to make prediction using the model got by training(any model it can read). Get and Fill the BDT score for each predicted trees then conserve this result.

NOTE: will tidy up this part 

#commenFuncs.py:
This file is mainly to get pandas-form data from TTree.
Here we can calculate weights and others variables using the info from TTree. 
We alse set class labels for training samples.




       


