import propagation as prop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import astropy.io.fits as pyfits
import control_matrix as cm
from general_formulas import *
from multiprocessing import Pool
import iterative_solver as it
import iterative_pytorch as it_pytorch
import forward_model_pytorch as prop_torch
import torch
import tqdm

path = '/home/ehold13/PDI_SIMULATIONS/'
# Load the data
# open look KL modes
OL_KL = pyfits.getdata(path+'turbulence_data/turbMod_down_07arcsec_50x5000.fits')
# plot the data
# plt.figure()
# plt.imshow(OL_KL/589, cmap='inferno',aspect='auto')
# plt.colorbar()
# plt.show()

# for each time iteration, run the phase retrieval
# define the parameters
pinhole_size = 0.685 #0.5
pup_width = 2**8
# want the same sampling of the pinhole for all cases
fp_oversamp = int(2**4/pinhole_size)
frac =  0.5 #0.2
method = 'KL'

if True:

    def rms_calcs(error):
        cnms = error/589
        # calculate the magnitude of the aberrations
        mag = np.sqrt(np.nansum(cnms**2))*589
        # run the forward model
        intensity = prop.propagate(cnms,frac,pinhole_size,max_zerns=np.shape(OL_KL)[0],pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589,mode_type=method)

        # now find the phase
        phase = it.iterative_solver(intensity,frac,pinhole_size,max_zerns=np.shape(OL_KL)[0],pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589,mode_type=method)

        rms = calc_rms(phase,cnms)*589

        return rms,mag
    
    rms, mag = []*len(OL_KL.T),[]*len(OL_KL.T)
    for i in tqdm.tqdm(range(len(OL_KL.T))):
        r,m = rms_calcs(OL_KL.T[i])
        m = m.item()
        print('Abberation Magnitude: {}, RMS: {}'.format(np.round(m,0),np.round(r,2)))
        rms.append(r)
        mag.append(m)
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.scatter(np.arange(len(rms)),rms,c='k',s=10,marker='o')
        ax2.scatter(np.arange(len(rms)),mag,c='r',s=1)
        ax1.set_xlabel('Time Iteration')
        ax1.set_ylabel('RMS (nm)')
        ax2.set_ylabel('Magnitude of Aberrations (nm)')
        plt.title('Pinhole Size = {}'.format(pinhole_size))
        # ask matplotlib for the plotted objects and their labels
        ax2.plot([],[],'.k',label='RMS')
        ax2.plot([],[],'.r',label='Turbulence Magnitude')
        ax2.legend()
        plt.savefig('figures/rms_mag_{}_iterative.png'.format(pinhole_size))
        plt.close()
    
        np.save('turbulence_data/rms_mag_{}_iterative'.format(pinhole_size),rms)

    # plot the rms and magnitude
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(np.arange(len(rms)),rms,c='k',s=5)
    ax2.scatter(np.arange(len(rms)),mag,c='r',s=1)
    ax1.set_xlabel('Time Iteration')
    ax1.set_ylabel('RMS (nm)')
    ax2.set_ylabel('Magnitude of Aberrations (nm)')
    plt.title('Pinhole Size = {}'.format(pinhole_size))
    # ask matplotlib for the plotted objects and their labels
    ax2.plot([],[],'.k',label='RMS')
    ax2.plot([],[],'.r',label='Turbulence Magnitude')
    ax2.legend()
    plt.savefig('figures/rms_mag_{}_iterative.png'.format(pinhole_size))
    plt.close()

if False:
    def rms_calc_pytorch(error):
        # change the data type to torch and divide by 589
        error /= 589
        error = error.astype(np.float64)
        cnms = torch.from_numpy(error)
        # calculate the magnitude of the aberrations
        mag = torch.sqrt(torch.nansum(cnms**2))*589
        # run the forward model
        
        intensity = prop_torch.forward_prop(max_zerns=np.shape(OL_KL)[0],
                                            cnms=cnms,
                                            pup_width=pup_width,
                                            fp_oversamp=fp_oversamp,
                                            wavelength=0.589,
                                            pinhole_size=pinhole_size,
                                            mode_type=method)

        # now find the phase
        phase = it_pytorch.iterative_solver(intensity,frac,pinhole_size,
                                            max_zerns=np.shape(OL_KL)[0],
                                            pup_width=pup_width,
                                            fp_oversamp=fp_oversamp,
                                            wavelength=0.589,
                                            mode_type=method)
        # if the phase is a tensor make it a numpy array
        if type(phase) == torch.Tensor:
            phase = phase.detach().numpy()
        if type(cnms) == torch.Tensor:
            cnms = cnms.detach().numpy()
        rms = calc_rms(phase,cnms)*589

        return rms,mag
    
    # plot the rms and magnitude
    rms, mag = []*len(OL_KL.T),[]*len(OL_KL.T)
    for i in tqdm.tqdm(range(len(OL_KL.T))):
        r,m = rms_calc_pytorch(OL_KL.T[i])
        m = m.item()
        print('Abberation Magnitude: {}, RMS: {}'.format(np.round(m,0),np.round(r,2)))
        rms.append(r)
        mag.append(m)
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.scatter(np.arange(len(rms)),rms,c='k',s=10,marker='o')
        ax2.scatter(np.arange(len(rms)),mag,c='r',s=1)
        ax1.set_xlabel('Time Iteration')
        ax1.set_ylabel('RMS (nm)')
        ax2.set_ylabel('Magnitude of Aberrations (nm)')
        plt.title('Pinhole Size = {}'.format(pinhole_size))
        # ask matplotlib for the plotted objects and their labels
        ax2.plot([],[],'.k',label='RMS')
        ax2.plot([],[],'.r',label='Turbulence Magnitude')
        ax2.legend()
        plt.savefig('figures/rms_mag_{}_iterative_pytorch_more.png'.format(pinhole_size))
        plt.close()
    
        np.save('turbulence_data/rms_mag_{}_iterative_pytorch_more'.format(pinhole_size),rms)

    # plot the rms and magnitude
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(np.arange(len(rms)),rms,c='k',s=5)
    ax2.scatter(np.arange(len(rms)),mag,c='r',s=1)
    ax1.set_xlabel('Time Iteration')
    ax1.set_ylabel('RMS (nm)')
    ax2.set_ylabel('Magnitude of Aberrations (nm)')
    plt.title('Pinhole Size = {}'.format(pinhole_size))
    # ask matplotlib for the plotted objects and their labels
    ax2.plot([],[],'.k',label='RMS')
    ax2.plot([],[],'.r',label='Turbulence Magnitude')
    ax2.legend()
    plt.savefig('figures/rms_mag_{}_iterative_pytorch_more.png'.format(pinhole_size))
    plt.close()

if False:
    # generate the control matrix
    IM_inv = cm.generate_matrix(max_zerns=np.shape(OL_KL)[0],pup_width=pup_width,fp_oversamp=fp_oversamp,pinhole_size=pinhole_size,wavelength=0.589,ratio=frac,amp=1e-4,mode=method)

    if False:
        def image_turb(t):
            cnms = OL_KL[:,t]/589
            # run the forward model
            intensity = prop.propagate(cnms,frac,pinhole_size,max_zerns=np.shape(OL_KL)[0],pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589)

            inten = plt.imshow(intensity,cmap='inferno')
            return inten

        # make the animation
        fig = plt.figure()
        imgs = []
        for t in range(np.shape(OL_KL)[1]):
            img = image_turb(t)
            imgs.append([img])

        ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True)
        plt.show()

    # generate the interferogram for each time step
    intensity_no_aberrations = prop.propagate(np.zeros(np.shape(OL_KL)[0]),frac,pinhole_size,max_zerns=np.shape(OL_KL)[0],pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589,mode_type=method)

    def rms_calcs(error):
        cnms = error/589
        # calculate the magnitude of the aberrations
        mag = np.sqrt(np.nansum(cnms**2))*589
        # run the forward model
        intensity = prop.propagate(cnms,frac,pinhole_size,max_zerns=np.shape(OL_KL)[0],pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589,mode_type=method)

        # now find the phase
        phase = np.dot(IM_inv,intensity.ravel()-intensity_no_aberrations.ravel())

        # calculate the rms
        phase[0] = 0
        
        rms = calc_rms(phase,cnms)*589

        return rms,mag

    with Pool(10) as p:
        vals = p.map(rms_calcs,OL_KL.T)
    rms, mag = zip(*vals)
    np.save('turbulence_data/rms_mag_{}_control_matrix'.format(pinhole_size),rms)
    # plot the rms and magnitude
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(np.arange(len(rms)),rms,c='k',s=1)
    ax2.scatter(np.arange(len(rms)),mag,c='r',s=1)
    ax1.set_xlabel('Time Iteration')
    ax1.set_ylabel('RMS (nm)')
    ax2.set_ylabel('Magnitude of Aberrations (nm)')
    plt.title('Pinhole Size = {}'.format(pinhole_size))
    # ask matplotlib for the plotted objects and their labels
    ax2.plot([],[],'.k',label='RMS')
    ax2.plot([],[],'.r',label='Turbulence Magnitude')
    ax2.legend()
    plt.savefig('figures/rms_mag_{}.png'.format(pinhole_size))
    plt.close()


