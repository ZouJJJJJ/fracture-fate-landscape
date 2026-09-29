# -*- coding: utf-8 -*-
"""Supplementary figures for WP0-WP4, modeled on the DiffusionOT paper panel set.
Outputs figS1..figS7 into wpN_out / a supp folder.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from sklearn.ensemble import RandomForestClassifier

BASE = ''+ROOT+''
SUPP = f'{BASE}/supp_figures'
import os; os.makedirs(SUPP, exist_ok=True)
stages = ['1','2','3','4']; stage_names=['E11.5','E12.5','E13.5','E14.5']
state_order=['mesenchyme_condensation','committed_chondro','fibrous_perichondrium','tendon','immune']
state_color=dict(zip(state_order,['#1f77b4','#2ca02c','#ff7f0e','#9467bd','#d62728']))

atlas=np.load(f'{BASE}/wp0_out/atlas.npz',allow_pickle=True)
Z2=atlas['Z2']; state=atlas['state']; cluster=atlas['cluster']; stg=atlas['stage']
fields=np.load(f'{BASE}/wp1_out/fields.npz',allow_pickle=True)
L=fields['L']; v2=fields['v2']; g=fields['g']; grid2=fields['grid2']
gx=fields['gx']; gy=fields['gy']; Dper=fields['D_per']; Dsc=float(fields['D_scalar'])
op=np.load(f'{BASE}/wp1_out/pca_operator.npz',allow_pickle=True)
latent=op['latent']; Q=op['Q']
ext=np.load(f'{BASE}/wp1_out/pca_operator_ext.npz',allow_pickle=True)
grn=np.load(f'{BASE}/wp3_out/grn_matrix.npz',allow_pickle=True)
A=grn['A']; genes=grn['genes']; traj_gene=grn['traj_gene']; growth_gene=grn['growth_gene']
trn=np.load(f'{BASE}/wp2_out/transitions.npz',allow_pickle=True)
class_mass=trn['class_mass']; Sidx=trn['S']; overall=trn['overall']; start_comp=trn['start_comp']
Sidx=np.vstack([np.array(r,int) for r in Sidx])
labs=['P','F','C','O']; lab_names=['progenitor','fibrous','chondrocyte','other']
fmap={'mesenchyme_condensation':'P','limb_mesenchyme':'P','fibrous_perichondrium':'F',
      'tendon':'F','committed_chondro':'C','proliferative_chondro':'C','immune':'O'}
cls=np.array([fmap.get(s,'O') for s in state])
li={l:i for i,l in enumerate(labs)}
Tf=[class_mass[t]/class_mass[t].sum(1,keepdims=True) for t in range(3)]

# ============ figS1 embedding & composition (WP0) ============
fig,ax=plt.subplots(2,2,figsize=(12,10))
for s in stages:
    k=stg==int(s)
    ax[0,0].scatter(Z2[k,0],Z2[k,1],s=4,label=stage_names[int(s)-1],alpha=.6)
ax[0,0].legend(markerscale=3); ax[0,0].set_title('A  2D embedding by stage')
for sname in state_order:
    k=state==sname
    ax[0,1].scatter(Z2[k,0],Z2[k,1],s=4,label=sname,color=state_color[sname],alpha=.6)
ax[0,1].legend(markerscale=3); ax[0,1].set_title('B  2D embedding by cell state')
ax[1,0].scatter(Z2[:,0],Z2[:,1],c=cluster,cmap='tab20',s=4)
ax[1,0].set_title('C  14 unsupervised clusters')
comp=pd.read_csv(f'{BASE}/wp0_out/composition.csv').set_index('row_0')
comp=comp[state_order]
x=np.arange(4); bottom=np.zeros(4)
for sname in state_order:
    ax[1,1].bar(x,comp[sname].values,bottom=bottom,color=state_color[sname],label=sname)
    bottom+=comp[sname].values
ax[1,1].set_xticks(x,stage_names); ax[1,1].set_ylim(0,1)
ax[1,1].set_title('D  State composition over development'); ax[1,1].legend(fontsize=8)
for a in ax.flat: a.set_xticks([]); a.set_yticks([]) if a not in [ax[1,1]] else None
plt.tight_layout(); plt.savefig(f'{SUPP}/figS1_embedding_composition.png',dpi=140); plt.close()

# ============ figS2 vector field & growth (WP1) ============
fig,ax=plt.subplots(2,2,figsize=(12,10))
t=3
Xg,Yg=np.meshgrid(gx,gy)
vv=v2[t].reshape(120,120,2)
xlim=np.percentile(fields['coords2'][:,0],[1,99]); ylim=np.percentile(fields['coords2'][:,1],[1,99])
# A velocity quiver over the dense core region
U,V=vv[:,:,0],vv[:,:,1]
k=stg==4
ax[0,0].scatter(Z2[k,0],Z2[k,1],s=3,color='lightgray')
ax[0,0].quiver(Xg[::4,::4],Yg[::4,::4],U[::4,::4],V[::4,::4],color='navy',
               scale=.09,width=.0035,alpha=.8)
ax[0,0].set_xlim(xlim); ax[0,0].set_ylim(ylim)
ax[0,0].set_title('A  Velocity field (E14.5, dense core)')
# B growth map over dense core
gg=g[t].reshape(120,120)
lo,hi=np.nanpercentile(gg,[2,98])
cf=ax[0,1].contourf(Xg,Yg,np.clip(gg,lo,hi),levels=20,cmap='viridis')
ax[0,1].scatter(Z2[k,0],Z2[k,1],s=2,color='white',alpha=.3)
ax[0,1].set_xlim(xlim); ax[0,1].set_ylim(ylim)
ax[0,1].set_title('B  Growth rate g (E14.5, dense core)'); plt.colorbar(cf,ax=ax[0,1],fraction=.046)
x=np.arange(4)
spd_mu=np.array([np.sqrt((v2[i]**2).sum(1)).mean() for i in range(4)])
mg=np.array([g[i].mean() for i in range(4)])
ax[1,0].bar(x-.18,spd_mu,.36,label='mean |velocity|')
ax2=ax[1,0].twinx(); ax2.bar(x+.18,-mg,.36,color='orange',label='-mean growth g')
ax[1,0].set_xticks(x,stage_names); ax[1,0].set_title('C  Mean velocity & growth per stage')
ax[1,0].legend(loc='upper left',fontsize=8); ax2.legend(loc='upper right',fontsize=8)
ax[1,1].bar(np.arange(8),Dper,color='steelblue',label='per-dimension residual D')
ax[1,1].axhline(Dsc,color='red',ls='--',label=f'model noise d={Dsc}')
ax[1,1].set_title('D  Diffusion by latent dimension'); ax[1,1].legend(fontsize=8)
for a in [ax[0,0],ax[0,1]]: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f'{SUPP}/figS2_vectorfield_growth.png',dpi=140); plt.close()

# ============ figS3 landscape + score arrows + cells (WP1) ============
L_CAP=6.0
fig,ax=plt.subplots(2,2,figsize=(12,10))
levels=np.linspace(1.4,L_CAP,22)
for idx2 in range(4):
    a=ax.flat[idx2]
    Lc=np.clip(L[idx2],None,L_CAP)
    cf=a.contourf(Xg,Yg,Lc,levels=levels,cmap='viridis_r',extend='max')
    dLdy,dLdx=np.gradient(Lc,gy,gx)
    sx,sy=-dLdx,-dLdy; mag=np.sqrt(sx**2+sy**2)+1e-9
    a.quiver(Xg[::6,::6],Yg[::6,::6],sx[::6,::6]/mag[::6,::6],sy[::6,::6]/mag[::6,::6],
             color='white',scale=32,width=.0026,alpha=.75)
    k=stg==idx2+1
    for sname in state_order:
        kk=k&(state==sname)
        a.scatter(Z2[kk,0],Z2[kk,1],s=3,color=state_color[sname],alpha=.6)
    a.set_title(f'{stage_names[idx2]}  L=-log p & score')
    a.set_xlim(xlim); a.set_ylim(ylim); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f'{SUPP}/figS3_landscape_score.png',dpi=140); plt.close()

# ============ figS4 stochastic trajectories (WP2) ============
rng=np.random.default_rng(3)
D=Dsc
def bridge(z0,z1,n=8):
    pts=[]; dt=1.0/n
    for u in range(n+1):
        m=z0+(z1-z0)*u/n
        if u>0: m=m+rng.normal(0,np.sqrt(2*D*dt),size=z0.shape)
        pts.append(m)
    return np.array(pts)
fig,ax=plt.subplots(2,2,figsize=(12,10))
# A forward ensemble emanating from the E11.5 centroid; nearest-target OT geodesic
start_cent=latent[Sidx[0]].mean(0)
for traj_i in range(160):
    z=start_cent+rng.normal(0,.06,size=8); ccur=li['P']; p2=[z@Q]
    for tt in range(3):
        pr=Tf[tt][ccur]; tcl=rng.choice(4,p=pr/pr.sum())
        cand=Sidx[tt+1][cls[Sidx[tt+1]]==labs[tcl]]
        pool=cand if len(cand) else Sidx[tt+1]
        i1=pool[np.argmin(((latent[pool]-z)**2).sum(1))]; z1=latent[i1]
        seg=bridge(z,z1); p2+=list(seg[1:]@Q); z=z1; ccur=tcl
    p2=np.array(p2); ax[0,0].plot(p2[:,0],p2[:,1],color='blue',lw=.5,alpha=.22)
meanpath=np.array([latent[Sidx[s]].mean(0) for s in range(4)])@Q
ax[0,0].plot(meanpath[:,0],meanpath[:,1],color='red',lw=2,label='mean ODE path')
ax[0,0].scatter([start_cent@Q][0][0],[start_cent@Q][0][1],color='black',zorder=5,s=30)
ax[0,0].legend(); ax[0,0].set_title('A  Forward stochastic trajectories (SDE blue, ODE red)')
ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
# B terminal fate bar
ax[0,1].bar(np.arange(4),overall,color=['#1f77b4','#ff7f0e','#2ca02c','#7f7f7f'])
ax[0,1].set_xticks(np.arange(4),lab_names,rotation=20)
ax[0,1].set_title('B  Terminal fate E11.5->E14.5')
# C backward ancestry stacked
anc=pd.read_csv(f'{BASE}/wp2_out/backward_ancestry.csv',index_col=0)
bottom=np.zeros(4)
for j,c in enumerate(anc.columns):
    ax[1,0].bar(np.arange(4),anc[c].values,bottom=bottom,label=c)
    bottom+=anc[c].values
ax[1,0].set_xticks(np.arange(4),anc.index,rotation=20); ax[1,0].set_ylim(0,1)
ax[1,0].set_title('C  Backward ancestry (E11.5 source by terminal state)'); ax[1,0].legend(fontsize=8)
# D composed forward matrix heatmap
Ff=pd.read_csv(f'{BASE}/wp2_out/forward_fate.csv',index_col=0)
im=ax[1,1].imshow(Ff.values,cmap='Blues',vmin=0,vmax=1)
ax[1,1].set_xticks(np.arange(4),Ff.columns,rotation=30,ha='right'); ax[1,1].set_yticks(np.arange(4),Ff.index)
for i in range(4):
    for j in range(4): ax[1,1].text(j,i,f'{Ff.values[i,j]:.2f}',ha='center',va='center',fontsize=8)
ax[1,1].set_title('D  Composed forward transition matrix')
plt.colorbar(im,ax=ax[1,1],fraction=.046)
plt.tight_layout(); plt.savefig(f'{SUPP}/figS4_stochastic_trajectories.png',dpi=140); plt.close()

# ============ figS5 conditions detail (WP2) ============
cmdf=pd.read_csv(f'{BASE}/wp2_out/condition_metrics.csv',index_col=0)
conds=['WT_Intact','WT_Ischemic','CD47_KO']
def row(metric,day): return [cmdf.loc[metric,f'{c}|{day}'] for c in conds]
fig,ax=plt.subplots(2,2,figsize=(12,9)); xx=np.arange(3); w=.38
for a,metric,title in [(ax[0,0],'commitment','Chondrogenic commitment'),
                      (ax[0,1],'fibrotic','Fibrotic fraction')]:
    a.bar(xx-w/2,row(metric,'day4'),w,label='day4'); a.bar(xx+w/2,row(metric,'day7'),w,label='day7')
    a.set_xticks(xx,conds,rotation=15); a.set_title(title); a.legend()
ax[1,0].bar(xx-w/2,row('inter_ischemic','day7'),w,label='fibroblastic intermediate')
ax[1,0].bar(xx+w/2,row('proliferation','day7'),w,label='proliferation')
ax[1,0].set_xticks(xx,conds,rotation=15); ax[1,0].set_title('Intermediate & proliferation (day7)'); ax[1,0].legend()
bs=pd.read_csv(f'{BASE}/wp2_out/block_score.csv',index_col=0)
ax[1,1].bar(xx,[bs.loc[f'{c}|day7'].iloc[0] for c in conds],color=['gray','#bcbd22','#d62728'])
ax[1,1].set_xticks(xx,conds,rotation=15); ax[1,1].set_title('Nonunion block score (day7)')
plt.tight_layout(); plt.savefig(f'{SUPP}/figS5_conditions_detail.png',dpi=140); plt.close()

# ============ figS6 GRN & perturbation dose (WP3) ============
curated=['Sox9','Sox5','Sox6','Runx2','Sp7','Col2a1','Acan','Col11a1','Col10a1','Col1a1','Col3a1','Bmp2','Cdk8','Map2k1']
cur=[c for c in curated if c in genes]; cidx=[list(genes).index(c) for c in cur]
# reconstructed standardized expression for curated genes (linear PCA decode), cheap
Jdec=ext['Jdec']  # 8 x 1876
Xcur=latent @ Jdec[:, cidx]  # 27264 x len(cur)
# residualize on stage one-hot + proliferation (Mki67 reconstructed) + intercept
gpos_ext={g:i for i,g in enumerate(genes)}
prolif = latent @ Jdec[:, gpos_ext['Mki67']] if 'Mki67' in gpos_ext else np.zeros(len(latent))
H=np.column_stack([(stg==s).astype(float) for s in range(1,5)]+[prolif,np.ones(len(latent))])
beta,_,_,_=np.linalg.lstsq(H,Xcur,rcond=None); R=Xcur-H@beta
Rs=R/R.std(0,keepdims=True)
pcor=np.corrcoef(Rs.T)
fig,ax=plt.subplots(2,3,figsize=(17,10))
vmax=np.nanpercentile(np.abs(pcor),95)
im=ax[0,0].imshow(pcor,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
ax[0,0].set_xticks(range(len(cur)),cur,rotation=90,fontsize=7); ax[0,0].set_yticks(range(len(cur)),cur,fontsize=7)
ax[0,0].set_title('A  Regulatory matrix (partial corr, stage/cycle corrected)'); plt.colorbar(im,ax=ax[0,0],fraction=.046)
# network from strongest partial correlations (top edges, signed)
G=nx.DiGraph(); thr=np.nanpercentile(np.abs(pcor),80)
for i,a in enumerate(cur):
    for j,b in enumerate(cur):
        if i!=j and abs(pcor[i,j])>=thr: G.add_edge(b,a,weight=pcor[i,j])
posg=nx.spring_layout(G,seed=1)
ecol=['red' if G[u][v]['weight']>0 else 'blue' for u,v in G.edges]
nx.draw(G,posg,ax=ax[0,1],node_color='lightyellow',node_size=400,arrows=True,edge_color=ecol,width=1.2)
nx.draw_networkx_labels(G,posg,ax=ax[0,1],font_size=7)
ax[0,1].set_title('B  GRN directed graph (red co-activated, blue anti)')
def is_noise(g): return g.startswith(('ERCC','ENSMUST','Gm','Hist','Rik','Rpl','Rps','MT-','mt-'))
def diverging(vals,axp,title,topn=8):
    v=vals.copy(); v[np.array([is_noise(g) for g in genes])]=0
    ip=np.argsort(v)[-topn:]; inp=np.argsort(v)[:topn]
    ys=np.concatenate([genes[inp][::-1],genes[ip]])
    bv=np.concatenate([v[inp][::-1],v[ip]])
    axp.barh(range(len(ys)),bv,color=['#1f77b4' if x<0 else '#d62728' for x in bv])
    axp.set_yticks(range(len(ys)),ys,fontsize=8); axp.set_title(title)
diverging(traj_gene,ax[0,2],'C  Trajectory critical genes')
diverging(growth_gene,ax[1,0],'D  Growth-related genes')
# dose-response via RF + linear latent shift
rf=RandomForestClassifier(n_estimators=120,random_state=0).fit(latent,cls)
Jenc=ext['Jenc']; start=latent[Sidx[0]]
def dose(gname):
    gi=list(genes).index(gname); ks=np.linspace(-3,3,13); out=[]
    for k in ks: out.append(rf.predict_proba(start+k*Jenc[gi]).mean(0))
    return ks,100*np.array(out)
def chondro_slope(gname):
    ks,d=dose(gname); cc=list(rf.classes_).index('C')
    return np.polyfit(ks,d[:,cc],1)[0]
# panel E: Cdk8 (inhibitor; KO raises chondro). panel F: strongest positive-slope agonist
slopes={g:chondro_slope(g) for g in cur}
agon=max([g for g in cur if g!='Cdk8'], key=lambda g: slopes[g])
for a,gname in [(ax[1,1],'Cdk8'),(ax[1,2],agon)]:
    ks,d=dose(gname)
    for c in range(4): a.plot(ks,d[:,c],marker='o',ms=3,label=rf.classes_[c])
    a.axvline(0,color='gray',ls='--'); a.set_title(f'Perturbation dose-response: {gname}')
    a.set_xlabel('std fold change (KO<0, OE>0)'); a.set_ylabel('fate %'); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{SUPP}/figS6_grn_perturbation.png',dpi=140); plt.close()
print('dose agonist selected:',agon,'Cdk8 slope',round(slopes['Cdk8'],2))

# ============ figS7 spatial detail (WP4) ============
sp=np.load(f'{BASE}/wp4_out/spot_scores.npz',allow_pickle=True)
cols=list(sp['cols']); ci_={c:i for i,c in enumerate(cols)}
def arr(label): return sp[label]
fig,ax=plt.subplots(2,3,figsize=(17,10))
mi=pd.read_csv(f'{BASE}/wp4_out/morans_I.csv',index_col=0)
mods=['chondro','fibrous','hyp','BMP','MAPK']; xx=np.arange(len(mods)); w=.38
ax[0,0].bar(xx-w/2,mi['Control'][mods],w,label='Control'); ax[0,0].bar(xx+w/2,mi['NF1'][mods],w,label='NF1')
ax[0,0].set_xticks(xx,mods,rotation=20); ax[0,0].set_title("A  Moran's I"); ax[0,0].legend()
nf=arr('NF1'); k=nf[:,ci_['in_t']]==1
ax[0,1].scatter(nf[k,ci_['BMP']],nf[k,ci_['MAPK']],s=5,alpha=.4)
bmp=nf[k,ci_['BMP']]; mpk=nf[k,ci_['MAPK']]
r=np.corrcoef(bmp,mpk)[0,1]; ax[0,1].set_title(f'B  NF1 BMP vs MAPK (r={r:.2f})')
ax[0,1].set_xlabel('BMP'); ax[0,1].set_ylabel('MAPK')
# junction vs non enrichment (within NF1 z)
fib=nf[:,ci_['fibrous']]; lin=nf[:,ci_['chondro']]+nf[:,ci_['hyp']]
jm=(fib>np.percentile(fib,60))&(lin>np.percentile(lin,60))&k
def z(v,m): return (v[m]-v[k].mean())/(v[k].std()+1e-9)
en=[z(nf[:,ci_['BMP']],jm).mean(),z(nf[:,ci_['BMP']],~jm&k).mean(),
    z(nf[:,ci_['MAPK']],jm).mean(),z(nf[:,ci_['MAPK']],~jm&k).mean()]
ax[0,2].bar(np.arange(4),en,color=['#d62728','#d62728','#1f77b4','#1f77b4'])
ax[0,2].set_xticks(range(4),['BMP junc','BMP rest','MAPK junc','MAPK rest'],rotation=20,fontsize=8)
ax[0,2].set_title('C  NF1 junction vs rest (within-slice z)')
for a,mm in [(ax[1,0],'prog'),(ax[1,1],'hyp'),(ax[1,2],'inter_med')]:
    sc=a.scatter(nf[k,ci_['x']],-nf[k,ci_['y']],c=nf[k,ci_[mm]],s=12,cmap='viridis')
    a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_title(f'NF1 {mm} map')
    plt.colorbar(sc,ax=a,fraction=.046)
plt.tight_layout(); plt.savefig(f'{SUPP}/figS7_spatial_detail.png',dpi=140); plt.close()
print('all supplementary figures saved to',SUPP)
