import propagation as prop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import astropy.io.fits as pyfits
import control_matrix as cm
from general_formulas import *
import tqdm

path = '/home/ehold13/PDI_SIMULATIONS/'
# Load the data
# open look KL modes
OL_KL = pyfits.getdata(path+'turbMod_down_07arcsec_50x5000.fits')
# plot the data
# plt.figure()
# plt.imshow(OL_KL/589, cmap='inferno',aspect='auto')
# plt.colorbar()
# plt.show()

# for each time iteration, run the phase retrieval
# define the parameters
pinhole_size = 0.685 #0.5
pup_width = 2**6
# want the same sampling of the pinhole for all cases
fp_oversamp = int(2**3/pinhole_size)
frac =  0.5 #0.2
method = 'KL'

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
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
for t in tqdm.tqdm(range(np.shape(OL_KL)[1])):
    cnms = OL_KL[:,t]/589
    # calculate the magnitude of the aberrations
    mag = np.sqrt(np.nansum(cnms**2))*589
    # run the forward model
    intensity = prop.propagate(cnms,frac,pinhole_size,max_zerns=np.shape(OL_KL)[0],pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589,mode_type=method)

    # now find the phase
    phase = np.dot(IM_inv,intensity.ravel()-intensity_no_aberrations.ravel())

    # calculate the rms
    phase[0] = 0
    
    rms = calc_rms(phase,cnms)*589

    if t==0:
        ax1.scatter([t],[rms],c='k',s=1,label='RMS')
        ax2.scatter([t],[mag],c='r',s=1,label='Magnitude')
    else:
        ax1.scatter([t],[rms],c='k',s=1)
        ax2.scatter([t],[mag],c='r',s=1)

ax1.set_xlabel('Time Iteration')
ax1.set_ylabel('RMS (nm)')
ax2.set_ylabel('Magnitude of Aberrations (nm)')
ax1.legend()
ax2.legend()
plt.show()

