#                                                                input                                                            output
#python3 analysis.py -c config.json -m xgboost -i /Users/xin/program/ML/ML_analysis/GG_ML/LGBM/XGboost/LowMass/output.root -s XGboost/HighMass/ -p all
#python3 analysis.py -c config.json -m xgboost -i /Users/xin/OneDrive/program/ML/ML_analysis/GG_ML/sample/backup/GG_MVA_all.root -s  Lightgbm/MidMass/ -p all
python3 analysis.py -c config.json -m lightgbm -i /Users/xin/OneDrive/program/ML/ML_analysis/GG_ML/sample/backup/GG_MVA_all.root -s  Lightgbm/MidMass/ -p all
#python3 analysis.py -c config.json -m lightgbm -i /Users/xin/OneDrive/program/ML/ML_analysis/GG_ML/sample/GG_MVA_combi.root -s  Lightgbm/LowMass/ -p all
