#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pylab as plt
a, b,are_a, c, d, are_d = np.load("roc_lightgbm.npy",allow_pickle=True)
a1, b1,area_train, c1, d1, area_test = np.load("roc_xgboost.npy",allow_pickle=True)
print(a)
print(a1)
plt.figure(figsize=(9, 9))
plt.plot(c, d, label='lightgbm test ROC curve (area = %0.2f)' % are_a)
plt.plot(c1, d1, label='xgboost test ROC curve (area = %0.2f)' % area_test)
plt.plot(a, b, label='lightgbm train ROC curve (area = %0.2f)' % are_d)
plt.plot(a1, b1, label='xgboost train ROC curve (area = %0.2f)' % area_train)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
# Plot the ROC curve
plt.savefig('ROC.png')
plt.close()
