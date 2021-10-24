import uproot3, ROOT, array, logging
ROOT.PyConfig.IgnoreCommandLineOptions = True
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import commenFuncs
import pickle as pk
from tqdm import tqdm
logger = logging.getLogger()

# Currently uproot doesn't support updating a root file. Use root_numpy as the solution. Will switch to it once it support

        #tree = rfile.Get(treeName)
        #newtree = tree.CloneTree()
        #newtree = ROOT.RDataFrame(treeName,rfile)

def predict(inFileName, predictList, modelFolder, trainVars, MLmethod, outTreeName,ifweight,noNegWei):
    # Input and output file
    rfile = ROOT.TFile.Open(inFileName)
    trees = commenFuncs.getTreeNames(inFileName, predictList)
    rfile_out1 = ROOT.TFile.Open(modelFolder + '/output.root', 'recreate')
    #rfile_out1 = ROOT.TFile.Open(modelFolder + '/'+treeName.split(";")[0]+'.root', 'recreate')

    # For retrieve input pandas arrays
    dataFile = uproot3.open(inFileName)
    all_ttrees = dict(dataFile.allitems(filterclass=lambda cls: issubclass(cls, uproot3.tree.TTreeMethods)))

    for treeName in tqdm(trees):
        # get the output tree
        outArray = None
        tree = rfile.Get(treeName)
        newtree = tree.CloneTree()

        logger.info("Start predicting tree %s", treeName)
        # get the np array for the given var 
        up_tree = all_ttrees[treeName.encode("utf-8")]
        
        pd_array = commenFuncs.loadData(up_tree, trainVars,ifweight,None,noNegWei)
        print("pd_array*****   ",pd_array.size)
        if pd_array.size == 0:
            outArray = np.array([0])
            array2tree(outArray, outTreeName, newtree)
            continue
        print("predict***************************")
        print(pd_array.head())
        pd_array.drop('weights', inplace=True, axis=1)  # drop weight var because we don't need it
        np_array = pd_array.to_numpy()

        # Switch method and get the prediction
        Y_prob = getPredictProb(pd_array, MLmethod, modelFolder)
        #print("********y_prob*******", Y_prob)

        num_classes = Y_prob.ndim
        # For num_classes 1 case, it's a 1D numpy array, we just use it as score.
        # for num_classes > 1 cases, it's a 2D array and we currently use (signal_score > max(list_bkg_score)) ? signal_score : 1 - max(list_bkg_score)) to get the combined score
        if num_classes > 1:
            signal_score = Y_prob[:, num_classes - 1]
            print("**************num_classes************   ",signal_score)
            bkg_score = Y_prob[:, 0:num_classes - 1]
            print("********signal_score*******", signal_score)
            print("********bkg_score*******", bkg_score)
            # 1 - max(list_bkg_score)
            bkg_best_score = bkg_score.max(1)
            # signal_score > bkg_best_score ? : signal_score : (1 - bkg_best_score)
            best_score = np.where(signal_score > bkg_best_score, signal_score,
                                  (signal_score * signal_score / bkg_best_score)) #
            outArray = best_score
        else:
            outArray = Y_prob

        #rfile_out1 = ROOT.TFile.Open(modelFolder + '/'+treeName.split(";")[0]+'.root', 'recreate')
        print("*********** ",treeName,"  *****   ",treeName.split(";")[0])
        tree = ROOT.TTree(treeName.split(";")[0],treeName.split(";")[0])
        array2tree(outArray, outTreeName, tree)
        #rfile_out1.Close()


def getPredictProb(inArray, method, modelFolder):
    if method == "lightgbm":
        modelDataFile = "%s/%s" % (modelFolder, "lightgbm.model")
        gbm = lgb.Booster(model_file=modelDataFile)
        print("***model***  ",gbm)
        Y_prob = gbm.predict(inArray, num_iteration=gbm.best_iteration)
    elif method == "xgboost":
        # load model
        f = open(modelFolder+'/xgboost_model.pickle', 'rb')
        model = pk.load(f); f.close();
        print("***model***  ",model)
        xgb_out = xgb.DMatrix(inArray)
        #Y_prob = model.predict(inArray)
        Y_prob = model.predict_proba(inArray)

    return Y_prob


def array2tree(outArray, outName, outTree):
    # Store the outArray as float
    outArrayType = outName + '/F'
    outHolder = array.array('f', [0])
    outBranch = outTree.Branch(outName, outHolder, outArrayType)
    arrayLen = outArray.shape[0]
    for i in range(arrayLen):
        outHolder[0] = outArray[i]
        outBranch.Fill()
    # need to set entry after filling a tree
    outTree.SetEntries(-1)
    # copy the tree into a new file
    outTree.Write()


#def array2tree(outArray, outName, outTree):
#    # Store the outArray as float
##    outTree.Define(outName,"for(auto w : outArray) return w;")
#    outTree.Snapshot(outName,"output.root")
