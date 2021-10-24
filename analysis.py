import json, io, argparse, os
import training, predicting, commenFuncs
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s|%(levelname)s] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', dest="config", action="store", help="the machine learning config file you want to use", default="config.json")
parser.add_argument('-i', dest="inputFile", action="store", help="input file name", default=None)
parser.add_argument(
    '-s',
    dest="StoreDir",
    action="store",
    help="When training, it will be the dir to store the output. When predicting, it will be the input dir to read the model got from training",
    default=None)
parser.add_argument(
    '-p',
    dest="process",
    action="store",
    help="the process you want to do, choose among 'training', 'predicting' and 'all'",
    default="training")
parser.add_argument(
    '-m',
    dest="method",
    action="store",
    help="the machine learning method, choose from lightgbm(default), xgboost",
    default="lightgbm")

args = parser.parse_args()


def main():
    inFileName = args.inputFile
    configFile = args.config
    modelFolder = args.StoreDir
    MLmethod = args.method
    MLprocess = args.process

    if inFileName is None:
        logging.critical("No input file given! Exit")
        return

    logging.info("Use input file: %s", inFileName)
    logging.info("The output/input dir of the trained model data: %s", modelFolder)
    logging.info("The ML method used: %s", MLmethod)

    logging.info("Load config file: %s", configFile)
    config = commenFuncs.readJson(configFile)

    # load ML parameters
    trainVars = config["variables"]
    if trainVars is None:
        logging.critical(
            "No Variables is set for training or predicting! Please set it using \'\"variables\": [list_of_vars]\' in config file. Program terminated"
        )
        return

    MLParas = config[MLmethod]
    if MLParas is None:
        logging.critical("No ML method named %s in config file! Please check! Program terminated" % MLmethod)
        return

    doTraining = (MLprocess == "training" or MLprocess == "all")
    doPredicting = (MLprocess == "predicting" or MLprocess == "all")
    if not (doTraining or doPredicting):
        logging.critical(
            "Unknown process! Please choose process among 'predicting', 'predicting' and 'all'! Program terminated")
        return

    # First, make sure our config is right!
    trainList = None
    if doTraining:
        trainList = config["train"]
        if trainList is None:
            logging.critical("No train tree list in the config file! Please check!")
            return

    if doPredicting:
        predictList = config["prediction"]
        if predictList is None:
            logging.critical("No predict tree list in the config file! Please check!")
            return

    # get the number of calsses
    num_class = len(config['train'])
    if num_class < 2:
        logging.critical(
            'Too small classes! You should have at least 2 classes to separate signal and background! Program terminated'
        )
        return
    elif num_class == 2:  # IF num class is 2, will use 1 for binary
        num_class = 1
    MLParas['num_class'] = num_class

  #  weightVar = config["Weights"]
    weightVar = [config["totweight"],config["lumiScaling"]]
    if weightVar is None:
        logging.warning(
            "Warning! There is no weight var set! All of the weight is set to 1! Please ensure your dataset has no weight!"
        )

    try:
        largestAllowWei = config["largestAllowWei"]
    except KeyError:
        largestAllowWei = None

    try:
        trainSize = config["trainSize"]
        if trainSize > 1 or trainSize <= 0:
            logging.warning(
                "The training dataset fraction should smaller than 1 and larger than 0! For now, use the default value 0.7"
            )
            trainSize = 0.7
        else:
            logging.info("The training dataset fraction is set to %f", trainSize)
    except KeyError:
        logging.info("No training dataset fraction is set. Use the default value 0.7")
        trainSize = 0.7

    # Now we do the actual process for training!
    ifweight = True; noNegWei = False
    #if doTraining:
    if doTraining:
        logging.info("Do Training!")
        # If no modelFolder, mkdir it
        if not os.path.exists(modelFolder):
            os.mkdir(modelFolder)
        if MLmethod == "lightgbm":
            logging.info("Train data using Method: lightgbm")
            ifweight = True;noNegWei = False
            pdData = commenFuncs.loadAllData(inFileName, trainList, trainVars,ifweight, weightVar, largestAllowWei, noNegWei)
            training.lgmTrain(pdData[0], MLParas, modelFolder, trainSize,pdData[1])
        elif MLmethod == "xgboost":
            logging.info("Train data using Method: xgboost")
            ifweight = True; noNegWei = True # can not use negative weight -->  delete negative weight or just set weight to 1?
            pdData = commenFuncs.loadAllData(inFileName, trainList, trainVars,ifweight, weightVar, largestAllowWei, noNegWei)
            training.xgbTrain(pdData[0], MLParas, modelFolder, trainSize,pdData[1])
        elif MLmethod == "catboost":
            logging.info("Train data using Method: catboost")
            ifweight = True
            pdData = commenFuncs.loadAllData(inFileName, trainList, trainVars,ifweight, weightVar, largestAllowWei, noNegWei)
            training.ctbTrain(pdData[0], MLParas, modelFolder, trainSize,pdData[1])
        else:
            pass  # Todo: add  neutral network machine learning method!
        logging.info("Training done!")

    # Do predicting if we select it!
    if doPredicting:
    #if False:
        logging.info("Start writing predicting branch into the input file!")
        outTreeName = config["out_branch"]
        if outTreeName is None:
            logging.info("No branch name specific for predicting in the config file! Please check!")
            return
        logging.info("The output predicting branch name is " + outTreeName)
        predictList = config["prediction"]
        predicting.predict(inFileName, predictList, modelFolder, trainVars, MLmethod, outTreeName,ifweight,noNegWei)
        logging.info("Prediction info writing done!")

    logging.info("All process done! Program close. Have a nice day!")
    return


if __name__ == '__main__':
    main()
