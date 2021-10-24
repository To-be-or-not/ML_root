#setupATLAS
#lsetup "lcgenv -p LCG_95a x86_64-slc6-gcc62-opt pip"
pip3 install --user numpy==1.17.3
pip3 install --user setuptools wheel scipy scikit-learn lightgbm xgboost uproot3 pandas matplotlib scikit-plot shap graphviz

mkdir result
