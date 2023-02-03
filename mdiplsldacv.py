import numpy as np
import torch
from mdipls import find_sig_mig,mdipls_tt,ldapinv
from sklearn import metrics
#from collections import Counter
import random

def cal_auc_auprc(label, scores):
    fpr, tpr, thresholds = metrics.roc_curve(label, scores)
    auc = metrics.auc(fpr, tpr)
    return auc

def gidx2iidx(indexyy,isoidx2geneidx):
    indexXt = []
    geneidx2isoidx = {}
    idx = 0
    isolen = 0
    for i in indexyy:
        indexXt += list(isoidx2geneidx[i])
        geneidx2isoidx[idx] = np.array(range(len(isoidx2geneidx[i])))+isolen
        idx += 1
        isolen += len(isoidx2geneidx[i])
    return np.array(indexXt),geneidx2isoidx

def readdata(Xcal,ycal,Xtcal,Xtiso2gene,gl):
    # read source domain data(gene)
    Xs = Xcal
    ys = ycal
    Xt = Xtcal      
    
    #plan2 for Xt(all MIGs)
    ret,allgenes,sizes = find_sig_mig(Xtiso2gene[:, 1])
    mig = ret['mig']
    pos_mig = []
    neg_mig = []
    for imig in mig:
        q = np.where(gl[:,0] == imig)[0][0]
        if gl[q,1] == 0:
            neg_mig.append(imig)
        else:
            pos_mig.append(imig)
    
    # get sig label: labled target samples
    sig = ret['sig']
    sigisoindex = []
    siglabel = []
    for isig in sig:
        sigisoindex.append(np.where(Xtiso2gene[:,1] == isig)[0][0])
        siglabel.append(gl[np.where(gl[:,0] == isig)[0][0],1])
    Xtl = Xt[sigisoindex,:]
    ytl = np.matrix(np.array(siglabel)).T
           
    #generate labeled target domain data Xtl(isoform of negative MIGs)
    migisoindex_n = []
    for imig in neg_mig:
        migidx = np.where(Xtiso2gene[:,1] == imig)[0]
        migisoindex_n += list(migidx) 
    ytl2 = np.matrix(np.zeros((len(migisoindex_n),1)))
    Xtl2 = Xt[migisoindex_n,:]
    
    #generate semi-labeled target domain data Xmiso(isoform of positive MIG)
    SEED = 1
    random.seed(SEED)
    random.shuffle(pos_mig)
    
    migisoindex_p = []
    g2i = []
    for imig in pos_mig:
        migidx = np.where(Xtiso2gene[:,1] == imig)[0]
        migisoindex_p += list(migidx) 
        k = np.where(allgenes==imig)[0][0]
        g2i.append(sizes[k])
    yts = np.matrix(np.ones((len(pos_mig),1)))
    Xmiso = Xt[migisoindex_p,:]

    ret = {}
    ret['Xs'] = Xs
    ret['Xt'] = Xt
    ret['Xtl'] = Xtl
    ret['Xtl2'] = Xtl2
    ret['Xmiso'] = Xmiso
    ret['ys'] = ys
    ret['ytl'] = ytl
    ret['ytl2'] = ytl2
    ret['yts'] =yts

    return (ret,g2i)

def index_arranege(X,y,Xt,iso2gene,isoidx2geneidx,order):
#%+++ Order: =1  sorted, default. For CV partition.
#%           =0  random. 
#%           =2  original order    
    if order == 1:
        indexyy = np.argsort(np.array(y.T))[0]
        X_arr = X[indexyy,:]
        y_arr = y[indexyy,:]
        indexXt_arr,geneidx2isoidx_arr = gidx2iidx(indexyy,isoidx2geneidx)
        Xt_arr = Xt[indexXt_arr,:]
        iso2gene_arr = iso2gene[indexXt_arr,:]
        
    elif order == 0:
        indexyy = np.array(torch.randperm(len(y)))
        X_arr=X[indexyy,:]
        y_arr=y[indexyy,:]
        indexXt_arr,geneidx2isoidx_arr = gidx2iidx(indexyy,isoidx2geneidx)
        Xt_arr = Xt[indexXt_arr,:]
        iso2gene_arr = iso2gene[indexXt_arr,:]
    
    return (X_arr,y_arr,geneidx2isoidx_arr,Xt_arr,iso2gene_arr)

def mdiplsldacv(X,y,Xt,iso2gene,geneidx2isoidx,gl,V,A,lmd,K):
#%+++ K-fold Cross-validation for PLS-LDA
#%+++ Input:  X: m x n  (Sample matrix)
#%            y: m x 1  (measured property)
#%            A: The maximal number of LVs for cross-validation
#%            K: fold. when K = m, it is leave-one-out CV
#%+++ Output: Structural data: CV

    # status variable:  1: Inf
    check=0
            
    A = min([X.shape[0]-np.ceil(len(y)/K), len(V), A])

    (Mx,Nx) = X.shape
    groups = np.array(range(Mx)) % K
    
    YR = np.matrix(np.zeros((Xt.shape[0],A)))
    
    checktesttk = []
    for group in range(K):
        
        calk = np.where(groups!=group)[0]
        caltk,_ = gidx2iidx(calk,geneidx2isoidx)
        testk = np.where(groups==group)[0]
        testtk,_ = gidx2iidx(testk,geneidx2isoidx)
        
        checktesttk += list(testtk)
        
        Xcal=X[calk,:]
        ycal=y[calk]
        Xtcal = Xt[caltk,:]
        Xtiso2gene_c = iso2gene[caltk,:]        
        
#        Xtest=X[testk,:]
#        ytest=y[testk]
        Xttest = Xt[testtk,:][:,V]
#        Xtiso2gene_t = iso2gene[testtk,:]  
        
        datacal,g2i_cal = readdata(Xcal,ycal,Xtcal,Xtiso2gene_c,gl)
#        datatest,g2i_test = readdata(Xtest,ytest,Xttest,Xtiso2gene_t,gl)

        X_cal,Y_cal,W,T,P,_ = mdipls_tt(datacal,g2i_cal,V,A,lmd)
        
        # Check data effectivness
        if np.sum(np.isnan(T)) + np.sum(np.isinf(T)) > 0:
            check=1
            break
        
        YR_temp = np.matrix(np.zeros((len(testtk),A)))            
        # Xttest = Xttest - X_mean
#        Xttest = Xttest - np.mean(Xttest, axis = 0)

        Xttest = np.column_stack((Xttest, np.ones((Xttest.shape[0], 1))))
        
        for j in range(1,A+1):
            # train model.        
            TT=T[:,0:j]
            C=ldapinv(TT,Y_cal,0)
            coef = np.vstack((W[:,0:j]*C[0:-1,0],C[-1,0]))
            # predict
            y_est = Xttest*coef[0:-1]+coef[-1]
#            print(y_est.shape)
#            print(YR_temp[:,j-1].shape)
            YR_temp[:,j-1] = y_est
            
        YR[testtk,:]= YR_temp 
            
#        print('The %dth fold for MDIPLS-LDA finished.'%(group+1))
        
    if len(list(set(checktesttk))) == Xt.shape[0]:
        print('target samples are all extracted')
    else:
        print('target samples are NOT all extracted')
        
    if check==0:
        # Original order
#        YR=YR[indexyy,:]
#        y=y[indexyy]
        error=[]
        for i in range(A):
            yi = []
            for j in range(len(y)):
                yij  = np.max(YR[geneidx2isoidx[j],i])/2+0.5
#                if yij >= 0:
#                    yij = 1
#                else:
#                    yij = 0
                yi.append(yij)
            yi = np.matrix(np.array(yi)).T
            error.append(cal_auc_auprc(y,yi))  
#            error.append(np.sum(yi!=y))       
        error=np.array(error)  
#        error=np.array(error)/Mx
        mincv = np.max(error)
#        index = np.where(error == mincv)[0][0]
        return mincv
        
        
