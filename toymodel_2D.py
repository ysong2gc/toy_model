import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def gaussian_2d(npix): #Use odd number!!!
    ###Creating square 2D grid and the Gaussian shape of the source. 
    ###npix is the pixel number in each axis. It is preferred to use an odd number. 
    x, y = np.meshgrid(np.linspace(-int(npix/2),int(npix/2),npix), np.linspace(-int(npix/2),int(npix/2),npix))
    dst = np.sqrt(x*x+y*y)

    # Intializing sigma and muu
    sigma = 5
    muu = 0.000

    # Calculating normalized 2D Gaussian array, to be used as the base for the following calculations. 
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    return(gauss)

flat_field = 20. ###Average noise photon counts level.
snr = 0.5   ###signal to noise ratio
npix = 71 ##No of pixels per axis, with 0.1deg/pixel gridding


def get_roi(npix, flat_field, snr):
    ###npix is the pixel number in each axis. It is preferred to use an odd number. 
    ###flat_field is the average noise/background photon counts level.
    ###snr is the signal_to_noise ratio. 
    gauss = gaussian_2d(npix)
    ##This below is to construct a simulated observations. 
    roi=np.zeros(gauss.shape)
    for i in range(71):
        for j in range(71):
            roi[i,j]=np.random.poisson(flat_field)*gauss[i,j]* snr +np.random.poisson(flat_field) 
    return(roi)

def get_model(npix,flat_field, snr):
    ###npix is the pixel number in each axis. It is preferred to use an odd number. 
    ###flat_field is the average noise/background photon counts level.
    ###snr is the signal_to_noise ratio. 
    gauss = gaussian_2d(npix)
    model = gauss * flat_field * snr + flat_field #Theoretical photon count in each pixel. 
    return(model)

def get_tsmap(npix,flat_field, snr):
    ###npix is the pixel number in each axis. It is preferred to use an odd number. 
    ###flat_field is the average noise/background photon counts level.
    ###snr is the signal_to_noise ratio. 
    gauss = gaussian_2d(npix)
    roi = get_roi(npix, flat_field, snr)
    model = get_model(npix, flat_field, snr)
    tsmap = np.zeros(gauss.shape)
    for i in range(71):
        for j in range(71):
            tsmap[i,j]=-2*(np.log(1+snr)*roi[i,j]-snr*model[i,j])###Calculating TS value pixel by pixel. 
    #This function returns the TS map as well as the TS value of the source
    #Estimated as the mean of the central 9 pixels. 
    return(tsmap, np.mean(tsmap[int(npix/2)-1:int(npix/2)+2,int(npix/2)-1:int(npix/2)+2])) 

def stack_tsmaps(npix, flat_field, snr, no_of_stacks):
    ###npix is the pixel number in each axis. It is preferred to use an odd number. 
    ###flat_field is the average noise/background photon counts level.
    ###snr is the signal_to_noise ratio. 
    ###no_of_stacks is the number of ROIs to be used in the stack. 
    gauss = gaussian_2d(npix)
    stacked_tsmap = np.zeros(gauss.shape)
    ts_values = np.array([])
    for i in range(no_of_stacks):
        tsmap, tsvalue = get_tsmap(npix,flat_field, snr)
        stacked_tsmap += tsmap
        ts_values = np.append(ts_values, tsvalue)
    return(stacked_tsmap, ts_values) ##This function returns the 
           
if __name__ == "__main__":
    gauss = gaussian_2d(npix)
    ##This below produces the series of stacked TS maps in the slides
    fig3,axs3=plt.subplots(ncols=5,nrows=1,figsize=(30,5))
    newshape=list(gauss.shape)
    newshape.append(5)
    stacked_tsmaps=np.zeros(tuple(newshape)) ##Create a 3D array for easier storing of the 5 stacked TS maps then plotting them. 
    for stack_i in range(5):
        stacked_tsmaps[:,:,stack_i],ts_values=stack_tsmaps(npix,flat_field, snr, (stack_i+1)*10)
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
    #Note that the array: ts_values already contain 50 TS values of this source populations. 

    no_co_added=np.arange(0,40)
    cumTS=np.array([])
    for i in range(len(no_co_added)):
        cumTS=np.append(cumTS, np.sum(ts_values[:i])) ##Stacking TS values
    fig4,axs4=plt.subplots(1,1,figsize=(5,4))
    axs4.plot(no_co_added, cumTS)
    axs4.set_xlabel('Co-Added ROIs')
    axs4.set_ylabel('TS Value')
    axs4.set_xlim(0,40)
    axs4.set_ylim(0,cumTS.max()+10)
    axs4.axhline(25, ls='--', color='b')

    fig4.savefig('cumTS.png')
