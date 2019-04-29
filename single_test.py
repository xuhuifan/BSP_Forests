import numpy as np
import scipy.io

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import invgamma
import scipy.stats as stats

# import pickle
import sys

# Particle Gibbs for BSP Forest Regression


from utility import *
from p_class import *

# first_arg = sys.argv[1]
# first_arg = 1

def BSP_Forest_fun():
    IterationTime = 40

    mTree = 50
    budget_val = 0.5
    testt = 0
    wi = 0
    run_detail(IterationTime, mTree, budget_val, testt, wi)

def run_detail(IterationTime, mTree, budget_val, testt, wi):
    particleNUm = 20
    dataNum = 500  # dataNum: 2000, 200

    dimNum = 5 # requie dimNum >5
    [xdata, ydata] = Friedman_function_gen(dimNum, dataNum)
    train_test_ratio = 0.5

    xdata_train, ydata_train, xdata_test, ydata_test, ydata_train_mean, dd, hyper_sigma_1, hyper_sigma_2, variance_hat = pre_process_data(xdata, ydata, train_test_ratio)

    mus = 0
    maxStage = 10
    add_dts_v = add_dts(mTree, mus, variance_hat, maxStage, budget_val, xdata_train, ydata_train, dimNum)

    y_train_final = np.zeros((IterationTime, len(ydata_train)))
    y_test_final = np.zeros((IterationTime, len(ydata_test)))

    true_seq = []
    RMAE_seq = []
    RMAE_train_seq = []
    for itei in range(IterationTime):
        # print('current iteration is: '+str(itei))f
        predicted_y_train = np.zeros((add_dts_v.mTree, len(ydata_train)))
        predicted_y_test = np.zeros((add_dts_v.mTree, len(ydata_test)))

        add_dts_v.updates(particleNUm, maxStage, xdata_train, ydata_train, dimNum, hyper_sigma_1, hyper_sigma_2)
        print('Iteration '+str(itei)+' finished. ')
        true_seq.append(add_dts_v.true_var)
        for mi in range(add_dts_v.mTree):
            predicted_y_train[mi] = add_dts_v.add_dts[mi].assign_to_data(len(add_dts_v.add_dts[mi].z_label) - 1)
            predicted_y_test[mi] = add_dts_v.add_dts[mi].predict_data(xdata_test)

            # add_dts_v.budgetSample(add_dts_v.add_dts[0].budgetseq[0])
        y_train_final[itei] = np.sum(predicted_y_train, axis=0)
        y_test_final[itei] = np.sum(predicted_y_test, axis=0)
        RMAE_seq.append(np.mean(abs(y_test_final[itei].reshape((-1)) - ydata_test)*dd))
        RMAE_train_seq.append(np.mean(abs(y_train_final[itei].reshape((-1)) - ydata_train)*dd))

    plt.plot(RMAE_train_seq)
    plt.plot(RMAE_seq)
    plt.legend(['Train', 'Test'])
    plt.title('BSP-Forest')
    plt.show()

if __name__ == '__main__':
    BSP_Forest_fun()
