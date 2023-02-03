# encoding:utf-8
import pandas  as pd
import numpy as np
import random
from collections import Counter
from sklearn import metrics
from sklearn.metrics import precision_recall_curve



###############
# sub-routines
def gidx2iidx(indexyy, isoidx2geneidx):
    indexXt = []
    geneidx2isoidx = {}
    idx = 0
    isolen = 0
    for i in indexyy:
        indexXt += list(isoidx2geneidx[i])
        geneidx2isoidx[idx] = np.array(range(len(isoidx2geneidx[i]))) + isolen
        idx += 1
        isolen += len(isoidx2geneidx[i])
    return np.array(indexXt), geneidx2isoidx


def readdata(Xcal, ycal, Xtcal, Xtiso2gene, gl):
    # read source domain data(gene)
    Xs = Xcal
    ys = ycal
    Xt = Xtcal

    # plan2 for Xt(all MIGs)
    ret, allgenes, sizes = find_sig_mig(Xtiso2gene[:, 1])
    mig = ret['mig']
    pos_mig = []
    neg_mig = []
    for imig in mig:
        q = np.where(gl[:, 0] == imig)[0][0]
        if gl[q, 1] == 0:
            neg_mig.append(imig)
        else:
            pos_mig.append(imig)

    # get sig label: labled target samples
    sig = ret['sig']
    sigisoindex = []
    siglabel = []
    for isig in sig:
        sigisoindex.append(np.where(Xtiso2gene[:, 1] == isig)[0][0])
        siglabel.append(gl[np.where(gl[:, 0] == isig)[0][0], 1])
    Xtl = Xt[sigisoindex, :]
    ytl = np.matrix(np.array(siglabel)).T

    # generate labeled target domain data Xtl(isoform of negative MIGs)
    migisoindex_n = []
    for imig in neg_mig:
        migidx = np.where(Xtiso2gene[:, 1] == imig)[0]
        migisoindex_n += list(migidx)
    ytl2 = np.matrix(np.zeros((len(migisoindex_n), 1)))
    Xtl2 = Xt[migisoindex_n, :]

    # generate semi-labeled target domain data Xmiso(isoform of positive MIG)
    SEED = 1
    random.seed(SEED)
    random.shuffle(pos_mig)

    migisoindex_p = []
    g2i = []
    for imig in pos_mig:
        migidx = np.where(Xtiso2gene[:, 1] == imig)[0]
        migisoindex_p += list(migidx)
        k = np.where(allgenes == imig)[0][0]
        g2i.append(sizes[k])
    yts = np.matrix(np.ones((len(pos_mig), 1)))
    Xmiso = Xt[migisoindex_p, :]

    ret = {}
    ret['Xs'] = Xs
    ret['Xt'] = Xt
    ret['Xtl'] = Xtl
    ret['Xtl2'] = Xtl2
    ret['Xmiso'] = Xmiso
    ret['ys'] = ys
    ret['ytl'] = ytl
    ret['ytl2'] = ytl2
    ret['yts'] = yts

    return (ret, g2i)


def index_arranege(X, y, Xt, iso2gene, isoidx2geneidx, order):
    # %+++ Order: =1  sorted, default. For CV partition.
    # %           =0  random.
    # %           =2  original order
    if order == 1:
        indexyy = np.argsort(np.array(y.T))[0]
        X_arr = X[indexyy, :]
        y_arr = y[indexyy, :]
        indexXt_arr, geneidx2isoidx_arr = gidx2iidx(indexyy, isoidx2geneidx)
        Xt_arr = Xt[indexXt_arr, :]
        iso2gene_arr = iso2gene[indexXt_arr, :]

    elif order == 0:
        indexyy = np.array(torch.randperm(len(y)))
        X_arr = X[indexyy, :]
        y_arr = y[indexyy, :]
        indexXt_arr, geneidx2isoidx_arr = gidx2iidx(indexyy, isoidx2geneidx)
        Xt_arr = Xt[indexXt_arr, :]
        iso2gene_arr = iso2gene[indexXt_arr, :]

    return (X_arr, y_arr, geneidx2isoidx_arr, Xt_arr, iso2gene_arr)

def mdipls(X, Y, Xs, Xt, Xtl, ytl, Xmiso, Xps, g2i, A, lmd):

    n = X.shape[0]
    pvar = X.shape[1]+1
    #
    ns = Xs.shape[0]
    nt = Xt.shape[0]

    #
    Xts = Xtl
    Yts = ytl

    #
    P = np.matrix(np.zeros((pvar, A)))
    W = np.matrix(np.zeros((pvar, A)))
    q = np.matrix(np.zeros((1, A)))
#    B = np.matrix(np.zeros((pvar, A)))

    #normalization
    # X_temp = X - np.mean(X, axis=0)
    # Y_temp = Y
    # Xs = Xs - np.mean(Xs, axis=0)
    # Xt = Xt - np.mean(Xt, axis=0)

    #add constant column
    X_temp = np.column_stack((X, np.ones((X.shape[0], 1))))
    Xs = np.column_stack((Xs, np.ones((Xs.shape[0], 1))))
    Xt = np.column_stack((Xt, np.ones((Xt.shape[0], 1))))

    #calculate column means of expression of positive SIG
    # Xpsmean = np.mean(Xps, axis=0)

    for i in range(A):
        part1 = Xs.T * Xs / (ns - 1)
        part2 = Xt.T * Xt / (nt - 1)

        # generate Xts (isoform expression value matrix) for positive MIGs
        if i == 0:
            select_mig_iso_ori = set(range(len(Xmiso)))
            select_mig_iso = set()
            count = 0
            while select_mig_iso != select_mig_iso_ori and count <= 100:

                select_mig_iso_ori = select_mig_iso
                select_mig_iso = set()
                # Xts_temp = Xts - np.mean(Xts,axis=0)
                Xts_temp = np.column_stack((Xts, np.ones((Xts.shape[0], 1))))
                # Yts_temp = Yts - np.mean(Yts)
                wt = (2*Y.T*X_temp+2*Yts.T*Xts_temp)*(((2*Y.T*Y)[0,0]*np.eye(pvar)+lmd*(part1 - part2)+((2*Yts.T*Yts)[0,0]*np.eye(pvar))).I)
                wt = wt/np.linalg.norm(wt)
                w = wt.T

                Xts = Xtl
                Yts = ytl

                start_idx = 0
                for g in range(len(g2i)):
                    v_compare = []
                    #find the isoform that might be mostly responsible for the positive label
                    for iso in range(g2i[g]):
                        obj_v_temp = abs(np.column_stack((Xmiso[iso+start_idx,:],np.matrix([1])))*w-np.mean(np.column_stack((Xps,np.ones((Xps.shape[0], 1))))*w))
                        v_compare.append(obj_v_temp)
                    min_index = v_compare.index(min(v_compare, key = abs))
                    selected_index = min_index+start_idx
                    Xts = np.vstack((Xts, Xmiso[selected_index,:]))
                    Yts = np.vstack((Yts,np.matrix(np.ones((1,1)))))
                    select_mig_iso.add(selected_index)
                    start_idx += g2i[g]
                count += 1

            X = np.vstack((X,Xts))
            Y = np.vstack((Y,Yts))

            # X_mean = np.mean(X, axis=0)
            # Y_mean = np.mean(Y)
            # X = X - X_mean
            X = np.column_stack((X, np.ones((X.shape[0], 1))))
            # Y = Y - Y_mean

            X_ori = X
            Y_ori = Y
            T = np.matrix(np.zeros((n+len(Yts), A)))

        y2 = Y.T * Y
        y2 = y2[0, 0]
        coef = lmd / (2 * y2)
        Q = np.matrix(np.eye(pvar)) + coef * (part1 - part2)

        #
        wt = Y.T * X * (Q.I) / y2
        w = wt.T
        w = w / np.linalg.norm(w)
        #
        t = X * w
        ts = Xs * w
        tt = Xt * w
        #
        pt = (t.T * t).I * t.T * X
        p = pt.T
        #
        pst = (ts.T * ts).I * ts.T * Xs
        ps = pst.T
        #
        ptt = (tt.T * tt).I * tt.T * Xt
        pt = ptt.T
        #
        qa = (t.T * t).I * Y.T * t
        qa = qa[0, 0]
        X = X - t * p.T
        Xs = Xs - ts * ps.T
        Xt = Xt - tt * pt.T
        Y = Y - qa * t
        # store
        T[:, i] = t
        P[:, i] = p
        W[:, i] = w
        q[:, i] = qa

#    for k in range(1, A + 1):
#        Wstar = W[:,0:k] * ((P[:,0:k].T * W[:,0:k]).I)
#        bk = Wstar * q[0:k].T
#        bk[-1] = bk[-1]+Y_mean- X_mean * bk[0:pvar-1]
#        B[:, k - 1] = bk
    #
    return (X_ori, Y_ori, W, T, P, q)

def mdipls_train(X, Y, Xs, Xt, Xtl, ytl, Xmiso, Xps, g2i, A, lmd):
    n = X.shape[0]
    pvar = X.shape[1] + 1
    #
    ns = Xs.shape[0]
    nt = Xt.shape[0]

    #
    Xts = Xtl
    Yts = ytl

    #
    P = np.matrix(np.zeros((pvar, A)))
    W = np.matrix(np.zeros((pvar, A)))
    q = np.matrix(np.zeros((1, A)))
    B = np.matrix(np.zeros((pvar, A)))

    # normalization
    # X_temp = X - np.mean(X, axis=0)
    # Y_temp = Y
    # Xs = Xs - np.mean(Xs, axis=0)
    # Xt = Xt - np.mean(Xt, axis=0)

    # add constant column
    X_temp = np.column_stack((X, np.ones((X.shape[0], 1))))
    Xs = np.column_stack((Xs, np.ones((Xs.shape[0], 1))))
    Xt = np.column_stack((Xt, np.ones((Xt.shape[0], 1))))

    # calculate column means of expression of positive SIG
    # Xpsmean = np.mean(Xps, axis=0)

    for i in range(A):
        part1 = Xs.T * Xs / (ns - 1)
        part2 = Xt.T * Xt / (nt - 1)

        # generate Xts (isoform expression value matrix) for positive MIGs
        if i == 0:
            select_mig_iso_ori = set(range(len(Xmiso)))
            select_mig_iso = set()
            count = 0
            while select_mig_iso != select_mig_iso_ori and count <= 100:

                select_mig_iso_ori = select_mig_iso
                select_mig_iso = set()
                # Xts_temp = Xts - np.mean(Xts,axis=0)
                Xts_temp = np.column_stack((Xts, np.ones((Xts.shape[0], 1))))
                # Yts_temp = Yts - np.mean(Yts)
                wt = (2 * Y.T * X_temp + 2 * Yts.T * Xts_temp) * (((2 * Y.T * Y)[0, 0] * np.eye(pvar) + lmd * (
                            part1 - part2) + ((2 * Yts.T * Yts)[0, 0] * np.eye(pvar))).I)
                wt = wt / np.linalg.norm(wt)
                w = wt.T

                Xts = Xtl
                Yts = ytl

                start_idx = 0
                for g in range(len(g2i)):
                    v_compare = []
                    # find the isoform that might be mostly responsible for the positive label
                    for iso in range(g2i[g]):
                        obj_v_temp = abs(np.column_stack((Xmiso[iso + start_idx, :], np.matrix([1]))) * w - np.mean(
                            np.column_stack((Xps, np.ones((Xps.shape[0], 1)))) * w))
                        v_compare.append(obj_v_temp)
                    min_index = v_compare.index(min(v_compare, key=abs))
                    selected_index = min_index + start_idx
                    Xts = np.vstack((Xts, Xmiso[selected_index, :]))
                    Yts = np.vstack((Yts, np.matrix(np.ones((1, 1)))))
                    select_mig_iso.add(selected_index)
                    start_idx += g2i[g]
                count += 1

            X = np.vstack((X, Xts))
            Y = np.vstack((Y, Yts))

            # X_mean = np.mean(X, axis=0)
            # Y_mean = np.mean(Y)
            # X = X - X_mean
            X = np.column_stack((X, np.ones((X.shape[0], 1))))
            # Y = Y - Y_mean

            T = np.matrix(np.zeros((n + len(Yts), A)))

        y2 = Y.T * Y
        y2 = y2[0, 0]
        coef = lmd / (2 * y2)
        Q = np.matrix(np.eye(pvar)) + coef * (part1 - part2)

        #
        wt = Y.T * X * (Q.I) / y2
        w = wt.T
        w = w / np.linalg.norm(w)
        #
        t = X * w
        ts = Xs * w
        tt = Xt * w
        #
        pt = (t.T * t).I * t.T * X
        p = pt.T
        #
        pst = (ts.T * ts).I * ts.T * Xs
        ps = pst.T
        #
        ptt = (tt.T * tt).I * tt.T * Xt
        pt = ptt.T
        #
        qa = (t.T * t).I * Y.T * t
        qa = qa[0, 0]
        X = X - t * p.T
        Xs = Xs - ts * ps.T
        Xt = Xt - tt * pt.T
        Y = Y - qa * t
        # store
        T[:, i] = t
        P[:, i] = p
        W[:, i] = w
        q[:, i] = qa

    for k in range(1, A + 1):
        Wstar = W[:,0:k] * ((P[:,0:k].T * W[:,0:k]).I)
        bk = Wstar * q[:,0:k].T
        # bk[-1] = bk[-1]+Y_mean- X_mean * bk[0:pvar-1]
        B[:, k - 1] = bk


    return (B)

def mdipls_pred(X, b):
    ypred = X * b

    return (ypred)

def find_sig_mig(genelist):
    counter = Counter(genelist)
    allgenes = np.array(list(counter.keys()))
    sizes = np.array(list(counter.values()))
    sig = list(allgenes[sizes == 1])
    mig = list(allgenes[sizes > 1])
    ret = {}
    ret['sig'] = sig
    ret['mig'] = mig
    return (ret,allgenes,sizes)

def cal_auc(label, scores):
    fpr, tpr, thresholds = metrics.roc_curve(label, scores)
    auc = metrics.auc(fpr, tpr)
    return (auc)

def cal_auc_auprc(label, scores):
    precision, recall, thresholds = precision_recall_curve(label, scores)
    auprc = metrics.auc(recall, precision)
    return (auprc)

def cal_auprc(label, scores):
    pos_score = list(scores[np.where(label == 1)[0]])
    pos_label = list(label[np.where(label == 1)[0]])

    neg_score = list(scores[np.where(label != 1)[0]])
    neg_label = list(label[np.where(label != 1)[0]])

    Neg_nums = len(neg_score)
    posi_nums = len(pos_score)

    try:
        Per = Neg_nums / posi_nums

        if Per <= 9:
            score = pos_score + neg_score
            label = pos_label + neg_label
            auprc = cal_auc_auprc(label, score)

        elif Per > 9:
            neg_choose = int(posi_nums * 9)
            itera = 100
            auprclist = []
            for i in range(itera):
                neg_choose_score = random.sample(neg_score, neg_choose)
                neg_choose_label = random.sample(neg_label, neg_choose)
                score = neg_choose_score + pos_score
                label = neg_choose_label + pos_label
                auprc_temp = cal_auc_auprc(label, score)
                auprclist.append(auprc_temp)
            auprc = np.mean(auprclist)

    except:
        auprc = np.nan
    return (auprc)

def model_eval(scores, iso2gene, posigene):
    niso = iso2gene.shape[0]
    genes = list(set(iso2gene[:, 1]))
    ngenes = len(genes)

    # AUC for sig
    tmp,_,_ = find_sig_mig(list(iso2gene[:, 1]))
    # sig = tmp['sig']
    mig = tmp['mig']

    # scores
    genescores = {}
    for i in range(niso):

        igene = iso2gene[i, 1]
        iscore = scores[i]
        if igene in genescores:
            if iscore > genescores[igene]:
                genescores[igene] = iscore
        else:
            genescores[igene] = iscore

    # prepare label and scores for AUC
    label = np.zeros(ngenes, dtype='int')
    scores = np.zeros(ngenes)
    geneclass = np.ones(ngenes)
    combined = np.zeros(shape=(ngenes, 3))
    i = 0

    rownames = []
    for igene in genescores.keys():
        scores[i] = genescores[igene]
        if igene in posigene:
            label[i] = 1
        if igene in mig:
            geneclass[i] = 2

        rownames.append(igene)
        i = i + 1
    # combine results
    combined[:, 0] = scores
    combined[:, 1] = label
    combined[:, 2] = geneclass

    ksig = (geneclass == 1)
    kmig = (geneclass == 2)

    scores_sig = combined[ksig, 0]
    label_sig = combined[ksig, 1]

    scores_mig = combined[kmig, 0]
    label_mig = combined[kmig, 1]

    combined = pd.DataFrame(combined, index=rownames)

    # AUC and AUPRC

    res0 = cal_auc(label, scores)
    res0_auprc = cal_auprc(label, scores)

    try:
        res1 = cal_auc(label_sig, scores_sig)
        res1_auprc = cal_auprc(label_sig, scores_sig)
    except:
        res1 = 0
        res1_auprc = 0

    try:
        res2 = cal_auc(label_mig, scores_mig)
        res2_auprc = cal_auprc(label_mig, scores_mig)
    except:
        res2 = 0
        res2_auprc = 0

    ret = {}
    ret['auc'] = round(res0, 4)
    ret['auprc'] = round(res0_auprc,4)
    ret['auc_sig'] = round(res1, 4)
    ret['auprc_sig'] = round(res1_auprc, 4)
    ret['auc_mig'] = round(res2, 4)
    ret['auprc_mig'] = round(res2_auprc, 4)
    ret['gene_score'] = combined
    return (ret)

def mdipls_tt(train,g2i,V0,A,lmd):
    # build a model
    if V0 != 'all':
        Xs = train['Xs'][:,V0] # source samples
        Xt = train['Xt'][:,V0]
        Xmiso = train['Xmiso'][:,V0]# target samples
        Xtl = train['Xtl'][:,V0]  # labelled target samples
        Xtl2 = train['Xtl2'][:,V0]  # labelled target samples
    else: 
        Xs = train['Xs']
        Xt = train['Xt']
        Xmiso = train['Xmiso'] 
        Xtl = train['Xtl']  # labelled target samples
        Xtl2 = train['Xtl2']  # labelled target samples
        
    ys = train['ys']
    yts = train['yts'] 
    ytl = train['ytl']
    ytl2 = train['ytl2']
    
    X = Xs # supervised manner
    y = ys  # supervised manner
    Xts = np.vstack((Xtl,Xtl2))  # supervised manner
    yts = np.vstack((ytl,ytl2))  # supervised manner
    # Xts = Xtl
    # yts = ytl
    
    k = np.where(ytl[:,0]==1)
    k = k[0]
    
    Xps = Xtl[k,:]

    return (mdipls(X, y, Xs, Xt, Xts, yts, Xmiso, Xps, g2i, A, lmd))

def mdipls_cv(train, V0, A, lmd, order=1, K = 3):
    # build a model
    Xs = train['Xs'][:, V0]  # source samples
    ys = train['ys']
    Xt = train['Xt'][:, V0]
    iso2gene = train['iso2gene']
    isoidx2geneidx = train['isoidx2geneidx']
    gl = train['gl']

    X, y, geneidx2isoidx, Xt, iso2gene = index_arranege(Xs, ys, Xt, iso2gene, isoidx2geneidx, order)

    A = min([X.shape[0] - np.ceil(len(y) / K), len(V0), A])
    (Mx, Nx) = X.shape
    groups = np.array(range(Mx)) % K
    YR = np.matrix(np.zeros((Xt.shape[0], A)))

    for group in range(K):

        calk = np.where(groups != group)[0]
        caltk, _ = gidx2iidx(calk, geneidx2isoidx)
        Xcal = X[calk, :]
        ycal = y[calk]
        Xtcal = Xt[caltk, :]
        Xtiso2gene_c = iso2gene[caltk, :]
        datacal, g2i_cal = readdata(Xcal, ycal, Xtcal, Xtiso2gene_c, gl) #read data_Cal from X_cal

        Xs_cal = datacal['Xs']
        Xt_cal = datacal['Xt']
        Xmiso_cal = datacal['Xmiso'] # target samples
        Xtl_cal = datacal['Xtl']  # labelled target samples
        Xtl2_cal = datacal['Xtl2']  # labelled target samples

        ys_cal = datacal['ys']
        ytl_cal = datacal['ytl']
        ytl2_cal = datacal['ytl2']

        X_cal = Xs_cal  # supervised manner
        y_cal = ys_cal  # supervised mannery_cal
        Xts_cal = np.vstack((Xtl_cal, Xtl2_cal))  # supervised manner
        yts_cal = np.vstack((ytl_cal, ytl2_cal))  # supervised manner

        k = np.where(ytl_cal[:, 0] == 1)
        k = k[0]
        Xps_cal = Xtl_cal[k, :]

        beta = mdipls_train(X_cal, y_cal, Xs_cal, Xt_cal, Xts_cal, yts_cal, Xmiso_cal, Xps_cal, g2i_cal, A, lmd)

        testk = np.where(groups == group)[0]
        testtk, _ = gidx2iidx(testk, geneidx2isoidx)
        Xttest = Xt[testtk, :]
        Xttest = np.column_stack((Xttest, np.ones((Xttest.shape[0], 1))))

        YR_temp = mdipls_pred(Xttest, beta)
        YR[testtk, :] = YR_temp

    auc = []
    for i in range(A):
        yi = []
        for j in range(len(y)):
            yij = np.max(YR[geneidx2isoidx[j], i])
            yi.append(yij)
        yi = np.matrix(np.array(yi)).T
        auc.append(cal_auc(y, yi))
    auc = np.max(auc)

    return (auc)

def read_data(isofile, genefile, labelfile):
    # read source domain data(gene)
    data = pd.read_csv(genefile, header=None, sep='\t')
    Xs = np.matrix(data.iloc[:, 1:])
    sgenes = data.values[:, 0]
    Xsnames = np.array(data.iloc[:, 0])
    ns = Xs.shape[0]
    nsample = Xs.shape[1]
    
    # get source response
    gl = pd.read_csv(labelfile, header=None, sep='\t').values
    ys = np.matrix(np.zeros((ns, 1)))
    i = 0
    for igene in sgenes:
        k = np.where(gl[:, 0] == igene)
        ys[i, 0] = gl[k, 1]
        i = i + 1

    # find posi genes
    kposi = np.where(gl[:, 1] == 1)
    posigene = list(gl[kposi][:, 0])
    
    # read target domain data Xt(isoform)
    data2 = pd.read_csv(isofile, header=None, sep='\t')
    list_custom = list(Xsnames)
    data2[1]=data2[1].astype('category')
    data2[1].cat.reorder_categories(list_custom, inplace=True)
    data2.sort_values(1, inplace=True)
    data2[0]=list(range(1,data2.shape[0]+1))
    data2.reset_index(drop=True, inplace=True)
        
        
    Xt = np.matrix(data2.iloc[:, 2:])
    Xtnames = np.array(data2.iloc[:, 0])
    iso2gene = data2.values[:, 0:2]    
    isoidx2geneidx={}
    for i in range(len(list_custom)):
        isoidx2geneidx[i] = np.where(data2[1] == list_custom[i])[0]
            
    #plan2 for Xt(all MIGs)
    ret,allgenes,sizes = find_sig_mig(iso2gene[:, 1])
    mig = ret['mig']
    nXt2 = 0
    for imig in mig:
        k = np.where(allgenes==imig)
        k = k[0]
        k = k[0]
        nXt2 += sizes[k]    
    Xt2 = np.matrix(np.zeros((nXt2, nsample)))
    pos_mig = []
    neg_mig = []
    Xt2names = []     
    j = 0
    for imig in mig:
        k = np.where(iso2gene[:, 1] == imig)
        k = k[0]
        for indx in k:
            Xt2[j, :] = Xt[indx, :]
            Xt2names.append(indx+1)
            j += 1           
        q = np.where(gl[:,0] == imig)[0][0]
        if gl[q,1] == 0:
            neg_mig.append(imig)
        else:
            pos_mig.append(imig)
    Xt2names = np.array(Xt2names) 

    # get sig label: labled target samples    
    sig = ret['sig']
    sigisoindex = []
    siglabel = []
    for isig in sig:
        sigisoindex.append(np.where(iso2gene[:,1] == isig)[0][0])
        siglabel.append(gl[np.where(gl[:,0] == isig)[0][0],1])
    Xtl = Xt[sigisoindex,:]
    ytl = np.matrix(np.array(siglabel)).T
        
    #generate labeled target domain data Xtl(isoform of negative MIGs)
    migisoindex_n = []
    for imig in neg_mig:
        migidx = np.where(iso2gene[:,1] == imig)[0]
        migisoindex_n += list(migidx) 
    ytl2 = np.matrix(np.zeros((len(migisoindex_n),1)))
    Xtl2 = Xt[migisoindex_n,:]  

    #generate semi-labeled target domain data Xmiso(isoform of positive MIG)
    
    migisoindex_p = []
    g2i = []
    for imig in pos_mig:
        migidx = np.where(iso2gene[:,1] == imig)[0]
        migisoindex_p += list(migidx) 
        k = np.where(allgenes==imig)[0][0]
        g2i.append(sizes[k])
    yts = np.matrix(np.ones((len(pos_mig),1)))
    Xmiso = Xt[migisoindex_p,:]

    ret = {}
    
    ret['Xs'] = Xs
    ret['Xt'] = Xt
    ret['Xt2'] = Xt2
    ret['Xtl'] = Xtl
    ret['Xtl2'] = Xtl2
    ret['Xmiso'] = Xmiso

    ret['ys'] = ys
    ret['ytl'] = ytl
    ret['ytl2'] = ytl2
    ret['yts'] =yts

    
    ret['Xsnames'] = Xsnames
    ret['Xtnames'] = Xtnames
    ret['Xt2names'] = Xt2names

    ret['posigene'] = posigene
    ret['iso2gene'] = iso2gene
    ret['isoidx2geneidx'] = isoidx2geneidx
    ret['gl'] = gl
    return (ret,g2i)

def target_project(X,b):
    w = b/np.linalg.norm(b)
    t = X * w
    p = X.T * t * (t.T * t).I
    Xtp = t * p.T
    Xr = X - Xtp
    # Compute selectivity ratio
    vart = []
    vartp = []
    varr =[]
    for i in range(np.size(X,1)):
        vart[i]=sum(sum(np.array(X[:,i])**2))
        vartp[i]=sum(sum(np.array(Xtp[:,i])**2))
        varr[i]=sum(sum(np.array(Xr[:,i])**2))
    sr=np.array(vartp)/(varr+np.spacing(1))
    return t,w,p,sr

def ldapinv(T,y,flag):
#flag: =1    Bayesian approximation.
#      =0    Fisher DA.
# The last element in C is the bias term.
    # y = y+Y_mean
    A = len(y)
    B = len(np.where(y[:,0]==1)[0])
    C = A-B
    r1=A/B
    r2=A/C
    kp = np.where(y[:,0]==1)[0]
    kn = np.where(y[:,0]==0)[0]
    TT = np.vstack((np.column_stack((T[kp,:],np.ones((B,1)))), (-1) * np.column_stack((T[kn,:],np.ones((C,1))))))    
    R = np.matrix(np.vstack((np.ones((B,1))*r1,np.ones((C,1))*r2)))
    BB = np.matrix(np.ones((A,1)))
    if flag == 1:
        C = (TT.T * TT).I * TT.T * BB
    elif flag==0:    
        try:
            C = (TT.T * TT).I * TT.T * R
        except:
            print('C is not invertible')
            C = np.linalg.pinv(TT.T * TT) * TT.T * R
    return C