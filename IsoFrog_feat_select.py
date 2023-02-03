import random
import numpy as np
from mdiplslda import mdiplslda
from mdiplsldacv import mdiplsldacv, index_arranege
import mdipls
import os
import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def weight2dict(varIndex, w):
    w_dict = {}
    for i in range(len(w)):
        w_dict[varIndex[i]] = w[i, 0]
    return w_dict


def random_weight(weight_data):
    total = sum(weight_data.values())  # 权重求和
    ra = random.uniform(0, total)  # 在0与权重和之前获取一个随机数
    curr_sum = 0
    ret = None
    keys = weight_data.keys()  # 使用Python3.x中的keys
    for k in keys:
        curr_sum += weight_data[k]  # 在遍历中，累加当前权重值
        if ra <= curr_sum:  # 当随机数<=当前权重和时，返回权重key
            ret = k
            break
    return ret


def randomsample(varIndex, Q, w):
    weight_data = weight2dict(varIndex, w)
    v = []
    for i in range(Q):
        q = random_weight(weight_data)
        v.append(q)
        weight_data.pop(q)
    return v


def generateNewModel(V0, nV1, coef, MDIPLSLDA, var):
    nV0 = len(V0)
    d = nV1 - nV0
    var_new = set(var)
    if d > 0:
        for i in V0:
            var_new -= set([i])
        var_new = list(var_new)
        kvar = list(range(len(var_new)))
        random.shuffle(kvar)
        perm = kvar[0:np.min([3 * d, len(kvar)])]



        V_new = [var_new[i] for i in perm]
        Vstartemp = V0 + V_new
        coef = MDIPLSLDA.LDA(Vstartemp)[0:-2, 0]
        coef = abs(coef) / sum(abs(coef))
        #      B=abs(MDIPLSLDA.coef_lda_origin[0:-1,0])
        index = np.argsort(-np.array(coef.T))[0]
        Vstar = [Vstartemp[idx] for idx in index[0:nV1]]
    elif d < 0:
        index = np.argsort(-np.array(coef.T))[0]
        Vstar = [V0[idx] for idx in index[0:nV1]]
    else:
        Vstar = V0
    return Vstar


def Rand_Frog(inputdir, GOtermname, A, lmd, N, Q, P, InitialOpt, K, order):
    # basic parameters of data input
    varIndex = list(range(P))
    MDIPLSLDA = mdiplslda(inputdir, GOtermname, A, lmd)
    data, g2i = MDIPLSLDA.readdata() #read training data from inputfile

    X = data['Xs']
    y = data['ys']
    Xt = data['Xt']
    iso2gene = data['iso2gene']
    isoidx2geneidx = data['isoidx2geneidx']
    gl = data['gl']

    X_arr, y_arr, geneidx2isoidx_arr, Xt_arr, iso2gene_arr = index_arranege(X, y, Xt, iso2gene, isoidx2geneidx, order)

    # Initialize a subset of variables: V0.
    if InitialOpt == 0:
        lp = list(range(P))
        random.shuffle(lp)
        V0 = lp[0:Q]
    else:
        #      MDIPLSLDA.LDA(varIndex)
        coef = MDIPLSLDA.LDA(varIndex)[0:-2, 0]
        w = abs(coef) / sum(abs(coef))
        V0 = randomsample(varIndex, Q, w)

    probability = np.zeros((1, P))
    RMSEP = []
    nVar = []
    V0_ori = []
    CV0 = 1000
    # Main loop for Random Frog
    for i in range(N):
        print('# of V0: ' + str(Q))

        # if i == 0:
        #     nVstar = 1735
        # else:
        nVstar = int(np.min([P, np.max([round((np.random.randn(1) * 0.3 * Q + Q)[0]), 2])]))
        #        if 1 < i <= 200:
        #            if nVstar < 50:
        #                nVstar = 50 + abs(round((np.random.randn(1)*0.3*100)[0]))
        if nVstar < 50:
            nVstar = int(50 + abs(round((np.random.randn(1) * 0.3 * 100)[0])))
        print('# of Vstar: ' + str(nVstar))
        #        MDIPLSLDA.LDA(V0)
        if V0 != V0_ori:
            coef = MDIPLSLDA.LDA(V0)[0:-2, 0]
            coef = abs(coef) / sum(abs(coef))
        #
        # if i == 0:
        #     Vstar = varIndex
        # else:
        Vstar = generateNewModel(V0, nVstar, coef, MDIPLSLDA, varIndex)

        if CV0 == 1000:
            CV0 = mdiplsldacv(X_arr, y_arr, Xt_arr, iso2gene_arr, geneidx2isoidx_arr, gl, V0, A, lmd, K)
        print('CV result for V0: '+str(CV0))
        if Vstar == V0:
            print('Vstar is the same with V0')
            continue
        else:
            CVstar = mdiplsldacv(X_arr, y_arr, Xt_arr, iso2gene_arr, geneidx2isoidx_arr, gl, Vstar, A, lmd, K)
            print('CV result for Vstar: ' + str(CVstar))

        if CVstar >= CV0:
            probAccept = 1
        else:
            probAccept = 0.02 * (CVstar + 0.001) / (CV0 + 0.001)

        randJudge = np.random.rand(1)

        V0_ori = V0

        if probAccept > randJudge:
            V0 = Vstar
            CV0 = CVstar
            RMSEP.append(CVstar)
            nVar.append(nVstar)
        else:
            V0 = V0
            CV0 = CV0
            RMSEP.append(CV0)
            nVar.append(Q)

        probability[0, V0] = probability[0, V0] + 1
        Q = len(V0)
        if i % 100 == 0:
            print('The %dth sampling for random frog finished.\n' % (i))

    probability = probability / N
    Vrank = np.argsort(-probability)[0]
    probability_sort = probability[0, Vrank]
    return (probability_sort, Vrank, nVar, RMSEP, data, g2i)


