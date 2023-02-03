# encoding:utf-8
import numpy as np
from mdipls import mdipls_cv
from IsoFrog_feat_select import Rand_Frog
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def SFS(inputdir, GOtermname, A, lmd, N, Q, P, InitialOpt, K, order):

    probability_sort, Vsort, nVar, RMSEP, data, g2i = Rand_Frog(inputdir, GOtermname, A, lmd, N, Q, P, InitialOpt, K, order)
    var_prop = [round(prob, 4) for prob in probability_sort]
    vars_sorted = [int(var) for var in Vsort]

    last_count = 0
    aucs = {}

    var_prop_set = list(set(var_prop))
    quantile = np.percentile(var_prop_set, list(np.arange(5,100,5)))
    quantile = quantile[::-1]

    sublist_1 = []
    sublist_2 = []

    for var in var_prop_set:
        if var >= quantile[0]:
            sublist_1.append(var)
    for var in var_prop_set:
        if quantile[1] <= var < quantile[0]:
            sublist_2.append(var)

    try:
        quintiles_1 = np.percentile(sublist_1, list(np.arange(20,100,20)))
        quintiles_2 = np.percentile(sublist_2, list(np.arange(20,100,20)))
    except:
        quintiles_1 = np.array([])
        quintiles_2 = np.array([])

    thresholds = np.sort(list(quantile) + list(quintiles_1) + list(quintiles_2) + [0])
    for threshold in thresholds:
        if threshold == 0:
            count = P
        else:
            count = np.sum(np.array(var_prop)>= threshold)

        if count == last_count:
            continue

        var = vars_sorted[0:count]
        var = np.array(var)

        try:
            auc = mdipls_cv(data, var, A, lmd)
        except:
            break

        last_count = count
        aucs[threshold] = (count, auc)

    auc_dict = aucs
    count_and_auc = list(auc_dict.values())
    auc_high = 0

    for i in range(len(count_and_auc)):
        if count_and_auc[i][1] >= auc_high:
            auc_high = count_and_auc[i][1]

    for i in auc_dict:
        if auc_dict[i][1] == auc_high:
            feat_select = vars_sorted[0:auc_dict[i][0]]
            break

    return (feat_select, data, g2i)
