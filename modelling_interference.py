import propagation
import forward_model as fm
import matplotlib.pyplot as plt
import numpy as np
import aotools

# want to simulate interference patterns to match 
# Chapter 6: Basic Interferometry and Optical Testing

# define the parameters
defocus = [-5,-2.5,0,2.5,5]
for amount in [7]:#np.linspace(6.5,10,8,dtype=np.float64,endpoint=True):
    zerns_to_check =[(0,amount),(5,amount),(7,amount),(6,amount),(10,amount)]
    names = ['Piston','Astigmatism','Horizontal Coma','Vertical Coma','Spherical']
    pup_width = 2**6
    for pair in zerns_to_check:
        cnms = np.zeros(16,dtype=np.float64)
        # initialise a 2x5 subplot
        fig,ax = plt.subplots(2,5,figsize=(10,5))
        # set the title
        fig.suptitle(f'{names[zerns_to_check.index(pair)]} Interference Patterns, {pair[1]}$\lambda$ RMS',fontsize=16)
        for Z in defocus:
            cnms = np.zeros(16,dtype=np.float64)
            cnms[3] = Z
            cnms[pair[0]] = pair[1]
            intensity = propagation.propagate(cnms,0.5,1,pup_width=pup_width,fp_oversamp=2**2,wavelength=10)

            final_aperture = aotools.circle(pup_width/2,pup_width)
            final_aperture = np.where(final_aperture == 0,np.nan,final_aperture)
            intensity = intensity * final_aperture
            ax[0,defocus.index(Z)].imshow(intensity,cmap='gray')
            #remove the axis labels
            ax[0,defocus.index(Z)].set_xticks([])
            ax[0,defocus.index(Z)].set_yticks([])
        for Z in defocus:
            cnms = np.zeros(16,dtype=np.float64)
            cnms[3] = Z
            cnms[1] = 5
            cnms[pair[0]] = pair[1]
            intensity = propagation.propagate(cnms,0.5,1,pup_width=pup_width,fp_oversamp=2**3,wavelength = 10)
            final_aperture = aotools.circle(pup_width/2,pup_width)
            final_aperture = np.where(final_aperture == 0,np.nan,final_aperture)
            intensity = intensity * final_aperture
            ax[1,defocus.index(Z)].imshow(intensity,cmap='gray')
            ax[1,defocus.index(Z)].set_xticks([])
            ax[1,defocus.index(Z)].set_yticks([])
    plt.show()