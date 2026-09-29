# -*- coding: utf-8 -*-
"""ITPS (Integrative Target Prioritization Strategy) funnel schematic."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np
BASE=''+ROOT+''
fig,ax=plt.subplots(figsize=(11,9)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')

def box(x,y,w,h,text,fc,fs=10,tc='black'):
    b=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',
                     fc=fc,ec='black',lw=1.1); ax.add_patch(b)
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=tc,wrap=True)

# title
ax.text(5,9.6,'Integrative Target Prioritization Strategy (ITPS)',ha='center',fontsize=14,weight='bold')
# four evidence inputs
inputs=[(0.3,7.6,'Pseudotime\ndrivers\n(endochondral axis)','#dbe9f6'),
        (2.75,7.6,'Failure-condition\nmediators\n(bootstrap-significant)','#fde8d4'),
        (5.2,7.6,'Fate-direction\nfrom OT coupling\n(up->inhibit, down->activate)','#e3f0d8'),
        (7.65,7.6,'Literature\ndruggability\n& existing evidence','#ece2f2')]
for x,y,t,fc in inputs: box(x,y,2.05,1.5,t,fc,9)
# arrows into funnel
for x,_,_,_ in inputs:
    ax.add_patch(FancyArrowPatch((x+1.02,7.55),(x+1.02,6.7),arrowstyle='-|>',mutation_scale=14,lw=1.2,color='gray'))
# funnel polygon
funnel=Polygon([(1.2,6.7),(8.8,6.7),(6.1,3.9),(3.9,3.9)],closed=True,fc='#f5f5f5',ec='black',lw=1.2)
ax.add_patch(funnel)
ax.text(5,5.6,'Weighted multi-evidence\nconvergence (composite score)',ha='center',va='center',fontsize=11)
# stem
ax.add_patch(FancyArrowPatch((5,3.9),(5,3.0),arrowstyle='-|>',mutation_scale=18,lw=2,color='black'))
# ranked list
box(3.0,1.3,4.0,1.6,'Ranked therapeutic targets\n1  CDK8 (inhibitor)   0.658\n2  SOX9 (agonist)      0.441\n3  BMP2 (agonist)     0.362\n4  MAP2K1 (inhibitor) 0.360','#fff6d6',9)
ax.add_patch(FancyArrowPatch((5,3.0),(5,2.95),arrowstyle='-|>',mutation_scale=12,lw=1.2))
# highlight CDK8
ax.text(5,0.7,'Top pan-non-union candidate: CDK8 inhibition',ha='center',fontsize=11,weight='bold',color='#b3261e')
plt.tight_layout(); plt.savefig(f'{BASE}/wp3_out/fig_itps.png',dpi=150,bbox_inches='tight'); plt.close()
print('ITPS saved')
