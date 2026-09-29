# -*- coding: utf-8 -*-
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import gzip, glob, os, gc
import numpy as np, pandas as pd
import scipy.sparse as sp

DD = ROOT + "/data/gse185940"
design=pd.read_csv(DD+"/plate_design.csv")
ab2row={r.ab:i for i,r in design.iterrows()}

def read_plate(f):
    df=pd.read_csv(f,sep="\t",index_col=0)
    genes=list(df.index.astype(str))
    W=df.values.T.astype(np.float32)  # cells x genes
    wmcs=list(df.columns)
    return wmcs,genes,W

# Pass1 union genes
union,seen,gcnt=[],set(),{}
for f in sorted(glob.glob(DD+"/GSM*.txt.gz")):
    _,genes,W=read_plate(f)
    pos=(W>0).sum(0)
    for g,c in zip(genes,pos):
        gcnt[g]=gcnt.get(g,0)+int(c)
        if g not in seen: seen.add(g);union.append(g)
genes=[g for g in union if gcnt.get(g,0)>=30]
gpos={g:i for i,g in enumerate(genes)}
print("kept genes",len(genes))

# Pass2 per plate normalize + map, collect global COO
ar,ac,ad=[],[],[]
meta=[]; offset=0
for f in sorted(glob.glob(DD+"/GSM*.txt.gz")):
    ab=os.path.basename(f).split("_")[1].replace(".txt.gz","")
    wmcs,fg,W=read_plate(f)
    lib=W.sum(1); keep=lib>0
    Wn=W[keep]/np.where(lib[keep]==0,1,lib[keep])[:,None]*1e4
    Wn=np.log1p(Wn).astype(np.float32)
    wmcs=[w for w,k in zip(wmcs,keep) if k]
    loc={g:j for j,g in enumerate(fg)}
    lj=np.array([loc[g] for g in genes],dtype=np.int64)
    sub=sp.csr_matrix(Wn[:,lj]).tocoo()
    ar.append(sub.row.astype(np.int64)+offset); ac.append(sub.col.astype(np.int64)); ad.append(sub.data)
    offset+=sub.shape[0]
    d=design.iloc[ab2row[ab]]
    for w in wmcs:
        meta.append((w,d.timepoint,d.selection if pd.notna(d.selection) else "NA",
                     d.treatment if pd.notna(d.treatment) else "None"))
    print(" ",ab,d.timepoint,sub.shape[0],flush=True)
    del W,Wn,sub; gc.collect()
X=sp.coo_matrix((np.concatenate(ad),(np.concatenate(ar),np.concatenate(ac))),
                shape=(offset,len(genes))).tocsr()
del ar,ac,ad; gc.collect()
mdf=pd.DataFrame(meta,columns=["wmc","timepoint","selection","treatment"])
print("merged",X.shape)
sp.save_npz(DD+"/gse185940_lognorm.npz",X)
pd.DataFrame({"gene":genes}).to_csv(DD+"/genelist.csv",index=False)
mdf.to_csv(DD+"/cell_meta.csv",index=False)
print(mdf.groupby(["timepoint","treatment"]).size())
