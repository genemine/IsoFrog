import numpy as np
from mdipls import ldapinv, read_data, mdipls_tt

class mdiplslda(object):
    
    def __init__(self,inputdir,GOtermname,A,lmd):
        self.inputdir = inputdir
        self.GOtermname = GOtermname
        self.A = A
        self.lmd = lmd
        self. isofile = inputdir + '/train_iso.tsv'
        self. genefile = inputdir + '/train_gene.tsv'
        self. labelfile = inputdir + '/train_label.label'
        
    def readdata(self):
        self.data, self.g2i = read_data(self.isofile, self.genefile, self.labelfile)
        return self.data, self.g2i
    
    def LDA(self,V0):
        check = 0
        X,y,W,T,P,q = mdipls_tt(self.data,self.g2i,V0,self.A,self.lmd)

        # Check data effectivness
        if np.sum(np.isnan(T)) + np.sum(np.isinf(T)) > 0:
            check=1
        if check==0:
            # target projection and selectivity ratio
#            tpt,tpw,tpp,SR=target_project(X,B[:,self.A-1])   
            C = ldapinv(T,y,0)                
            self.coef_lda_origin=np.vstack((W*C[0:-1,0],C[-1,0]))
            
        return self.coef_lda_origin
