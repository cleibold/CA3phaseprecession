import numpy as np
import pickle
import os
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.colors import LinearSegmentedColormap as owncmap
from matplotlib.colors import hsv_to_rgb as hsv2rgb
from utils import rootpth, hex2rgb

##
##

runstr='output/'
figstr='figures/'
#
#
# analysis parameters
#
DC_values=np.array([0])
Psi_values=np.array([0])
ph_shift=np.array([np.pi])
pct_around_median = 85;
#
#
#
#


#########
IDCS=['MECX','MECC','DGX','DGC']

PklDir = rootpth+runstr
FigDir = rootpth+figstr;

COLOR = {"gray":hex2rgb('666766'),
         "dgcontrol":hex2rgb('7DADCC'),
         "dglesion":hex2rgb('374C93'),
         "meccontrol":hex2rgb('E3B768'),
         "meclesion":hex2rgb('B54B48')}

if __name__ == "__main__":
   
    psi=Psi_values[0]
    
    for dc_comp in DC_values:
        for npf, phase_shift in enumerate(ph_shift):
            axA={}
            axP={}  
            figmec=plt.figure(figsize=(8,3))
            axA['MEC']=figmec.add_subplot(3,2,1)
            axP['MEC']=figmec.add_subplot(3,2,2)
            figdg=plt.figure(figsize=(8,3))
            axA['DG']=figdg.add_subplot(3,2,1)
            axP['DG']=figdg.add_subplot(3,2,2)

            for REGION_IDX in IDCS:
                l_key=REGION_IDX+'_'+str(dc_comp)

                
                file_name = 'Masked_'+REGION_IDX+'-PCT='+str(pct_around_median)+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"+'-phase_shift'+str(int(phase_shift/np.pi*180))
                load_name = os.path.join(PklDir, file_name + '.pkl')



                
                with open(load_name,'rb') as fd:
                    masked=pickle.load(fd)
                    
                    slopes=masked[REGION_IDX]['slope']
                    r2=masked[REGION_IDX]['r2']
                    phi0=masked[REGION_IDX]['phi0']
                    phif=masked[REGION_IDX]['phif']
                    pA=masked[REGION_IDX]['p(A)']
                    pPhi=masked[REGION_IDX]['p(phi)']
                    data=masked[REGION_IDX]['data']

                
                if 'MEC' in REGION_IDX:
                    FIG_TITLE = 'MEC'
                    fig=figmec
                    r_id="mec"
                else:
                    FIG_TITLE = 'DG'
                    fig=figdg
                    r_id="dg"
                    
                if 'X' in REGION_IDX:
                    linspecs = dict(color=COLOR[r_id+"lesion"], lw=2)
                else:
                    linspecs = dict(color=COLOR[r_id+"control"], lw=1)
        
                
                psm=np.convolve(pPhi[1]/np.nansum(pPhi[1]),np.ones(5)/5,'same')
                axP[FIG_TITLE].plot(pPhi[0],psm, **linspecs)
                axP[FIG_TITLE].set_xlabel('$\Phi$')
                axP[FIG_TITLE].set_ylabel('Rel. freq.')

                psm=np.convolve(pA[1]/np.nansum(pA[1]),np.ones(5)/5,'same')
                axA[FIG_TITLE].plot(pA[0], psm, **linspecs)
                axA[FIG_TITLE].set_xlabel('A')
                axA[FIG_TITLE].set_ylabel('Rel. freq.')
                
            figmec.subplots_adjust(wspace=0.4,hspace=0.3)
            figdg.subplots_adjust(wspace=0.4,hspace=0.3)
            figmec.suptitle('MEC', fontsize=12)
            figdg.suptitle('DG', fontsize=12)
            
            file_name = 'DG_masked-PCT='+str(pct_around_median)+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"+'-phase_shift'+str(int(phase_shift/np.pi*180))
            save_name = os.path.join(FigDir, file_name + '.eps')
            figdg.savefig(save_name)
            file_name = 'MEC_masked-PCT='+str(pct_around_median)+'-psi='+str(psi)+'-dc_comp='+f"{dc_comp:.2f}"+'-phase_shift'+str(int(phase_shift/np.pi*180))
            save_name = os.path.join(FigDir, file_name + '.eps')
            figmec.savefig(save_name)

    plt.close('all')
    plt.show(block=0)
        
            
