import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



###Creating 2D grid
x, y = np.meshgrid(np.linspace(-35,35,71), np.linspace(-35,35,71))
dst = np.sqrt(x*x+y*y)
  
# Intializing sigma and muu
sigma = 5
muu = 0.000
  
# Calculating normalized 2D Gaussian array, to be used as the base for the following calculations. 
gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )

flat_field = 20. ###Average noise photon counts level.
snr = 0.5   ###signal to noise ratio



def get_roi(flat_field, snr):
    ##This below is how to construct a simulated observations. 
    roi=np.zeros(gauss.shape)
    for i in range(71):
        for j in range(71):
            roi[i,j]=np.random.poisson(flat_field)*gauss[i,j]* snr +np.random.poisson(flat_field) 
    return(roi)

def get_model(flat_field, snr):
    model = gauss * flat_field * snr + flat_field #Theoretical photon count in each pixel. 
    return(model)

##This below produces the series of stacked TS maps in the slides
fig3,axs3=plt.subplots(ncols=5,nrows=1,figsize=(30,5))
newshape=list(gauss.shape)
newshape.append(5)
stacked_tsmaps=np.zeros(tuple(newshape)) ##Create a 3D array for easier storing of the 5 stacked TS maps then plotting them. 
for stack_i in range(5):
    no_of_stacks=(stack_i+1)*10
    for ijkijk in range(no_of_stacks):
        tsmap=np.zeros(gauss.shape)
        roi=np.zeros(gauss.shape)
        model=np.zeros(gauss.shape)
        for i in range(71):
            for j in range(71):
                model[i,j]=flat_field*gauss[i,j]*snr+flat_field
                roi[i,j]=np.random.poisson(flat_field)*gauss[i,j]*snr+np.random.poisson(flat_field)

        for i in range(71):
            for j in range(71):
                tsmap[i,j]=-2*(np.log(1+snr)*roi[i,j]-snr*model[i,j])###Calculating TS value pixel by pixel. 
        stacked_tsmaps[:,:,stack_i]=stacked_tsmaps[:,:,stack_i]+tsmap
vmin=stacked_tsmaps.min()
vmax=stacked_tsmaps.max()
for i in range(len(axs3.flat)):
    ax=axs3.flat[i]
    im=ax.imshow(stacked_tsmaps[:,:,i],vmax=vmax)
    ax.set_title('Stacked TS map,'+str((i+1)*10)+'ROIs, SNR = '+str(snr))
    if i==0:
        ax.set_ylabel(r'Dec Offset',fontsize=16)
    ax.set_yticks([5,15,25,35,45,55,65])
    ax.set_yticklabels([r'$3^{\circ}$',r'$2^{\circ}$',r'$1^{\circ}$',0,r'$-1^{\circ}$',r'$-2^{\circ}$',r'$-3^{\circ}$'],fontsize=16)
    ax.set_xlabel(r'R.A. Offset',fontsize=16)
    ax.set_xticks([5,15,25,35,45,55,65])
    ax.set_xticklabels([r'$3^{\circ}$',r'$2^{\circ}$',r'$1^{\circ}$',0,r'$-1^{\circ}$',r'$-2^{\circ}$',r'$-3^{\circ}$'],fontsize=16)

fig3.colorbar(im,ax=axs3.ravel().tolist(), label='TS')

fig3.savefig('stacked_tsmaps.png')

#We can also replicated the cumulative TS value increasing w/ stacked ROIs from literature as shown in the slides
ts_roi=np.array([]) ##TS value of the central source
#print(snr)
for nroi in range(40):
    tsmap=np.zeros(gauss.shape)
    roi=np.zeros(gauss.shape)
    for i in range(71):
        for j in range(71):
            model[i,j]=flat_field*gauss[i,j]*snr+flat_field
            roi[i,j]=np.random.poisson(flat_field)*gauss[i,j]*snr+np.random.poisson(flat_field)

    for i in range(71):
        for j in range(71):
            tsmap[i,j]=-2*(np.log(1+snr)*roi[i,j]-snr*model[i,j])
    ts_roi=np.append(ts_roi, np.average(tsmap[34:37,34:37])) ##TS value here, is calculated as the average of the central 9 pixels. 

    
no_co_added=np.arange(0,40)
cumTS=np.array([])
for i in range(len(no_co_added)):
    cumTS=np.append(cumTS, np.sum(ts_roi[:i])) ##Stacking TS values
fig4,axs4=plt.subplots(1,1,figsize=(5,4))
axs4.plot(no_co_added, cumTS)
axs4.set_xlabel('Co-Added ROIs')
axs4.set_ylabel('TS Value')
axs4.set_xlim(0,40)
axs4.set_ylim(0,cumTS.max()+10)
axs4.axhline(25, ls='--', color='b')

fig4.savefig('cumTS.png')
