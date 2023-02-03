import pandas as pd
import os
from IsoFrog_train_test import IsoFrog_train_test



A = int(20)
lmd = float(0.01)
N = 2000
Q = 200  # 50
P = 1735
InitialOpt = 0
order = 1
K = 2

# input data
inputdir='data/GO_demo'
outdir='output_traintest/'

if not os.path.exists(outdir):
    os.makedirs(outdir)


# MAIN
term=inputdir.split('/')[-1]

ret = IsoFrog_train_test(inputdir, term, A, lmd, N, Q, P, InitialOpt, K, order)

outfile=outdir+'/'+term+'.isoresolve.auc.auprc'
iso_scorefile=outdir+'/'+term+'.isoresolve.iso.score'
gene_scorefile=outdir+'/'+term+'.isoresolve.gene.score'
gene_scorefile=outdir+'/'+term+'.isoresolve.gene.score'

f=open(outfile,'w')

f.write('AUC\t'+str(ret['auc'])+'\n')
f.write('AUC_sig\t'+str(ret['auc_sig'])+'\n')
f.write('AUC_mig\t'+str(ret['auc_mig'])+'\n')
f.write('AUPRC\t'+str(ret['auprc'])+'\n')
f.write('AUPRC_sig\t'+str(ret['auprc_sig'])+'\n')
f.write('AUPRC_mig\t'+str(ret['auprc_mig'])+'\n')
f.write('Aopt\t'+str(ret['Aopt'])+'\n')
f.write('AUC_nlv\t'+str(ret['auc_nlv'])+'\n')
f.close()

print('AUC:'+str(ret['auc']))

isoscore= pd.DataFrame(ret['isoscore'])
isoscore.to_csv(iso_scorefile,index=None,header=None,sep='\t')


genescore= pd.DataFrame(ret['gene_score'])
genescore.to_csv(gene_scorefile,header=None,sep='\t')