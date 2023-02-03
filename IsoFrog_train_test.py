import numpy as np
from recursive_feature_elmination import RFE
from mdipls import mdipls_train,model_eval, mdipls_pred, read_data

def IsoFrog_train_test(inputdir, GOtermname, A, lmd, N, Q, P, InitialOpt, K, order):

    #TRAIN
    feat_select, train_data, g2i = RFE(inputdir, GOtermname, A, lmd, N, Q, P, InitialOpt, K, order)

    Xs = train_data['Xs'][:, feat_select]  # source samples
    Xt = train_data['Xt'][:, feat_select]   # target samples
    Xmiso = train_data['Xmiso'][:, feat_select]  # positive MIG target samples
    Xtl = train_data['Xtl'][:, feat_select] # SIG target samples
    Xtl2 = train_data['Xtl2'][:, feat_select]  # negative MIG target samples

    ys = train_data['ys']   # source labels
    ytl = train_data['ytl'] # SIG target labels
    ytl2 = train_data['ytl2']  # negative MIG target labels

    X = Xs
    y = ys
    Xts = np.vstack((Xtl, Xtl2))
    yts = np.vstack((ytl, ytl2))

    k = np.where(ytl[:, 0] == 1)
    k = k[0]
    Xps = Xtl[k, :] #positive SIG target samples

    beta = mdipls_train(X, y, Xs, Xt, Xts, yts, Xmiso, Xps, g2i, A, lmd)

    #TEST
    testisofile = inputdir + '/test_iso.tsv'
    testgenefile = inputdir + '/test_gene.tsv'
    testlabelfile = inputdir + '/test_label.label'

    test_data = read_data(testisofile, testgenefile, testlabelfile)
    Xttest = test_data['Xt'][:, feat_select]
    iso2gene = test_data['iso2gene']
    posigene = test_data['posigene']

    Xttest = np.column_stack(Xttest, np.ones((Xttest.shape[0], 1)))

    #make prediction
    predscore = mdipls_pred(Xttest, beta)
    predscore = 1 / (1 + np.exp(-predscore))

    # AUC
    result = {}
    result['auc'] = 0
    auc_nlv = []

    for i in range(A):
        result_i = model_eval(predscore[:, i], iso2gene, posigene)
        pscore_i = np.column_stack((iso2gene, predscore[:, i]))
        result_i['isoscore'] = pscore_i
        auc_nlv.append(result_i['auc'])
        if result_i['auc'] > result['auc']:
            result = result_i
            result['Aopt'] = i + 1

    result['auc_nlv'] = auc_nlv

    return (result)



