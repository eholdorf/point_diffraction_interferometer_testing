import numpy as np
import control_matrix as cm
import matplotlib.pyplot as plt
import propagation as prop
import aotools
from matplotlib.ticker import MultipleLocator
from fractions import Fraction
import aotools
from progress.bar import Bar
from multiprocessing import Pool
from general_formulas import *
import matplotlib.animation as animation
import iterative_solver as im
import tqdm

def RMSE_all_modes(pup_width, fp_oversamp,pinhole_size):
    """
    Calculate and plot the Root Mean Squared Error (RMSE) for different Zernike modes with fixed number of pixels for all.

    Parameters:
    - pup_width (float): Width of the pupil.
    - fp_oversamp (int): Oversampling factor for the focal plane.

    Returns:
    None

    This function generates a matrix, iterates over different Zernike modes, calculates
    the RMSE for each mode, and plots the results.

    Note: This code assumes the existence of the following external functions and modules:
    - cm.generate_matrix(pup_width, fp_oversamp)
    - prop.propagate(cnms, frac, pinhole_size, max_zerns, pup_width, fp_oversamp, wavelength, mode_type)
    - aotools.zernikeArray(max_zerns, pup_width, norm)
    - aotools.make_kl(max_zerns, pup_width, ri)
    - aotools.circle(radius, size)

    """
    frac = 0.5 # 0.5*pinhole_size
    # Generate the control matrix
    CM = cm.generate_matrix(pup_width=pup_width, fp_oversamp=fp_oversamp,pinhole_size=pinhole_size,ratio=frac)

    # Define parameters
    max_zerns = np.shape(CM)[0]
    wavelength = 0.589
    pup_width = int(np.shape(CM)[1] ** 0.5)

    # Generate aperture
    aperture = aotools.circle(pup_width/2, pup_width)
    aperture = np.where(aperture == 0, np.nan, aperture)

    # Generate flat propagated wave
    cnms_flat = np.zeros(max_zerns, dtype=np.float32)
    intensity_flat = prop.propagate(cnms_flat, frac, pinhole_size, max_zerns=max_zerns,
                                    pup_width=pup_width, fp_oversamp=fp_oversamp, wavelength=wavelength)

    # Generate Zernike modes
    amps = np.linspace(-0.5, 0.5, 100)
    modes = 'Zernike'

    scatter_colors = plt.cm.get_cmap('tab' + str(max_zerns), max_zerns)

    # Plot RMSE for each Zernike mode
    # plt.figure()
    for i in range(max_zerns):
        rms = []
        for a in amps:
            cnms = np.zeros(max_zerns, dtype=np.float32)
            cnms[i] = a
            intensity = prop.propagate(cnms, frac, pinhole_size, max_zerns=max_zerns,
                                       pup_width=pup_width, fp_oversamp=fp_oversamp, wavelength=wavelength, mode_type=modes)
            C = np.dot(CM, (intensity - intensity_flat).ravel())

            # Retrieve and compare true and retrieved waves
            if modes == 'Zernike':
                # Set the piston to 0
                C[0] = 0
                retrieved = np.einsum("ijk,i->jk", aotools.zernikeArray(max_zerns, pup_width, norm='rms'), C)
                true = np.einsum("ijk,i->jk", aotools.zernikeArray(max_zerns, pup_width, norm='rms'), cnms)
            elif modes == 'KL':
                # Set the piston to 0
                C[0] = 0
                retrieved = np.einsum("ijk,i->jk", aotools.make_kl(max_zerns, pup_width, ri=1e-4)[0], C)
                true = np.einsum("ijk,i->jk", aotools.make_kl(max_zerns, pup_width, ri=1e-4)[0], cnms)

            # Calculate RMSE
            difference = retrieved - true
            rmse = calc_rms(C, cnms)*589
            rms.append(rmse)

        # Plot RMSE for the current Zernike mode
        plt.scatter(amps, rms, label=f"Noll: {i+1}", color=scatter_colors(i))

    # Show legend and adjust plot
    plt.legend()
    ax = plt.gca()
    #ax.yaxis.set_major_locator(MultipleLocator(0.1))
    #ax.set_yticklabels([f'{Fraction(t * 10).limit_denominator()}$\lambda$/10' for t in ax.get_yticks()])
    if modes == 'Zernike':
        ax.set_xlabel("Zernike Amplitude")
    elif modes == 'KL':
        ax.set_xlabel("KL Amplitude")
    ax.set_ylabel("RMS Error (nm)")
    plt.show()

    return pup_width,fp_oversamp,pinhole_size,rms

def RMSE_one_config(pup_widths, fp_oversamps,cnms):
    """
    Calculate and plot the Root Mean Squared Error (RMSE) for same Zernike mode 
    with differing resolutions of the simulation.

    Parameters:
    - pup_widths (list): List of pupil widths.
    - fp_oversamps (list): List of oversampling factors for the focal plane.
    - cnms (numpy.ndarray): Numpy array representing the wavefront coefficients.

    Returns:
    None

    This function iterates over different pupil widths and oversampling factors,
    calculates the RMSE for each configuration, and plots the results.

    Note: This code assumes the existence of the following external functions and modules:
    - cm.generate_matrix(pup_width, fp_oversamp)
    - prop.propagate(cnms, frac, pinhole_size, max_zerns, pup_width, fp_oversamp, wavelength, mode_type)
    - aotools.zernikeArray(max_zerns, pup_width, norm)
    - aotools.make_kl(max_zerns, pup_width, ri)

    """
   # Iterate over pupil widths
    with Bar('Processing',fill='|',max = len(pup_widths)*len(fp_oversamps) ,suffix='%(percent).1f%% - %(eta)ds') as bar:
        for pup_width in pup_widths:
            rms = []
            # Iterate over oversampling factors
            for fp_oversamp in fp_oversamps:
                # Generate matrix
                CM = cm.generate_matrix(pup_width=pup_width, fp_oversamp=fp_oversamp, max_zerns=np.shape(cnms)[0])

                # Define parameters
                frac = 0.5
                pinhole_size = 1
                max_zerns = np.shape(CM)[0]
                wavelength = 0.589
                pup_width = int(np.shape(CM)[1] ** 0.5)

                # Generate flat propagated wave
                cnms_flat = np.zeros(max_zerns, dtype=np.float32)
                intensity_flat = prop.propagate(cnms_flat, frac, pinhole_size, max_zerns=max_zerns,
                                                pup_width=pup_width, fp_oversamp=fp_oversamp, wavelength=wavelength)

                # Generate wave with specified coefficients
                modes = 'Zernike'  # or 'KL'
                intensity = prop.propagate(cnms, frac, pinhole_size, max_zerns=max_zerns,
                                        pup_width=pup_width, fp_oversamp=fp_oversamp, wavelength=wavelength, mode_type=modes)

                # Calculate the difference in intensity
                C = np.dot(CM, (intensity - intensity_flat).ravel())

                # Set piston to 0
                C[0] = 0
                # Retrieve wavefront and true wavefront
                if modes == 'Zernike':
                    retrieved = np.einsum("ijk,i->jk", aotools.zernikeArray(max_zerns, pup_width, norm='rms'), C)
                    true = np.einsum("ijk,i->jk", aotools.zernikeArray(max_zerns, pup_width, norm='rms'), cnms)
                elif modes == 'KL':
                    retrieved = np.einsum("ijk,i->jk", aotools.make_kl(max_zerns, pup_width, ri=1e-4)[0], C)
                    true = np.einsum("ijk,i->jk", aotools.make_kl(max_zerns, pup_width, ri=1e-4)[0], cnms)

                # Calculate RMSE
                difference = retrieved - true
                rmse = np.nanstd(difference) / (2 * np.pi)  # * wavelength*1e3
                rms.append(rmse)

                bar.next()

            # Plot the results for the current pupil width
            #plt.scatter(fp_oversamps, rms, label=f"pup_width: {pup_width}")

        # Show legend and adjust plot
        # plt.legend()
        # ax = plt.gca()
        # ax.yaxis.set_major_locator(MultipleLocator(0.01))
        # ax.set_yticklabels([f'{Fraction(t * 100).limit_denominator()}$\lambda$/100' for t in ax.get_yticks()])
        # ax.set_xlabel("Focal Plane Oversampling Factor")
        # ax.set_ylabel("RMS Error")
        # plt.show()
    return pup_width,fp_oversamp,rms

def response_curve(pup_width, fp_oversamp,pinhole_size,show = False):
    """
    Calculate and plot the response curve for a given configuration.

    Parameters:
    - pup_width (float): Width of the pupil.
    - fp_oversamp (int): Oversampling factor for the focal plane.

    Returns:
    None

    This function generates a matrix, iterates over different Zernike modes, calculates
    the RMSE for each mode, and plots the results.

    Note: This code assumes the existence of the following external functions and modules:
    - cm.generate_matrix(pup_width, fp_oversamp)
    - prop.propagate(cnms, frac, pinhole_size, max_zerns, pup_width, fp_oversamp, wavelength, mode_type)
    - aotools.zernikeArray(max_zerns, pup_width, norm)
    - aotools.make_kl(max_zerns, pup_width, ri)
    - aotools.circle(radius, size)

    """
    
    frac = 0.5# pinhole_size**2
    # set phase amplitudes to test
    amps = np.linspace(-2,2,100)
    # generate the interferogram with no aberrations
    cnms = np.zeros(16,dtype=np.float64)
    intensity_flat = prop.propagate(cnms,frac,pinhole_size,max_zerns=16,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589)
    #iterate over modes and plot in subplot grid
    fig,axs = plt.subplots(4,4,figsize=(6,6))
    # generate a line for each of the axes
    for modes in tqdm.tqdm(range(16)):
        Cs = np.zeros(len(amps),dtype=np.float32)
        cnmss = np.zeros(len(amps),dtype=np.float32)
        for j,amp in enumerate(amps):
            cnms = np.zeros(16,dtype=np.float32)
            cnms[modes] = amp
            intensity = prop.propagate(cnms,frac,pinhole_size,max_zerns=16,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589)
            # generate the control matrix
            CM = cm.generate_matrix(pup_width=pup_width,fp_oversamp=fp_oversamp,max_zerns=16,pinhole_size=pinhole_size,ratio=frac)
            # retreived cnms
            C = np.dot(CM,intensity.ravel()-intensity_flat.ravel())
            #C = np.dot(CM,intensity.ravel())
            C[0]=0
            Cs[j] = C[modes]
            cnmss[j] = cnms[modes]
            # plot the result
            
        axs.flatten()[modes].plot(cnmss, Cs, 'ko')
    if show:
        plt.show()
    else:
        # save the figure
        plt.tight_layout()
        plt.savefig('response_curve_changing_p/response_curve_{}.png'.format(pinhole_size),dpi=300)
        plt.close()

def response_curve_iterative(pup_width, fp_oversamp,pinhole_size,show = False):
    """
    Calculate and plot the response curve for a given configuration.

    Parameters:
    - pup_width (float): Width of the pupil.
    - fp_oversamp (int): Oversampling factor for the focal plane.

    Returns:
    None

    This function generates a matrix, iterates over different Zernike modes, calculates
    the RMSE for each mode, and plots the results.

    Note: This code assumes the existence of the following external functions and modules:
    - cm.generate_matrix(pup_width, fp_oversamp)
    - prop.propagate(cnms, frac, pinhole_size, max_zerns, pup_width, fp_oversamp, wavelength, mode_type)
    - aotools.zernikeArray(max_zerns, pup_width, norm)
    - aotools.make_kl(max_zerns, pup_width, ri)
    - aotools.circle(radius, size)

    """
    
    frac = 0.5
    # set phase amplitudes to test
    amps = np.linspace(-2,2,100)
    # generate the interferogram with no aberrations
    cnms = np.zeros(16,dtype=np.float64)
    intensity_flat = prop.propagate(cnms,frac,pinhole_size,max_zerns=16,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589)
    #iterate over modes and plot in subplot grid
    fig,axs = plt.subplots(4,4,figsize=(6,6))
    # generate a line for each of the axes
    for modes in tqdm.tqdm(range(16)):
        Cs = np.zeros(len(amps),dtype=np.float32)
        cnmss = np.zeros(len(amps),dtype=np.float32)
        for j,amp in enumerate(amps):
            cnms = np.zeros(16,dtype=np.float32)
            cnms[modes] = amp
            intensity = prop.propagate(cnms,frac,pinhole_size,max_zerns=16,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=0.589)
            # use the iterative method to retrieve the wavefront
            C = im.iterative_solver(intensity,frac, pinhole_size, 16,0.589,pup_width,fp_oversamp,'Zernike')

            Cs[j] = C[modes]
            cnmss[j] = cnms[modes]
            # plot the result
            
        axs.flatten()[modes].plot(cnmss, Cs, 'ko')
    if show:
        plt.show()
    else:
        # save the figure
        plt.tight_layout()
        plt.title('Response Curve for Pinhole Size = {}, using Iterative Method'.format(pinhole_size))
        plt.savefig('response_curve_changing_p/response_curve_iterative_{}.png'.format(pinhole_size),dpi=300)
        plt.close()
# test the functions for a range of pup_width and fp_oversamp
if __name__=="__main__":

    if False:
        with Pool(10) as p:
            r = p.starmap(RMSE_all_modes,[(2**6,int(1/0.1)*2**2,0.1),(2**6,int(1/0.2)*2**2,0.2),
                                                             (2**6,int(1/0.3)*2**2,0.3)])#,(2**6,int(1/0.4)*2**2,0.4),
                                                             #(2**6,int(1/0.5)*2**2,0.5),(2**6,int(1/0.6)*2**2,0.6), 
                                                             #(2**6,int(1/0.7)*2**2,0.7),(2**6,int(1/0.8)*2**2,0.8), 
                                                             #(2**6,int(1/0.9)*2**2,0.9),(2**6,int(1/1.0)*2**2,1.0)])
        pup,samp,pinhole,rms = zip(*r)    
        plt.figure()
        plt.scatter(np.arange(len(rms)),rms)
        plt.show()


    if False:
        p = 0.685
        RMSE_all_modes(2**6,2**3*int(1/p),p)

    if True:
        p = 0.685
        #response_curve(2**9, int(2**3/p),p,show = True)
        response_curve_iterative(2**9, int(2**3/p),p,show = True)
    
    if False:
        p = np.linspace(0.1,0.5,6,endpoint=True)
        if True:
            imgs = []
            # use a progress bar with estimated time to completion
            with Pool(6) as mp:
                pts = 2**6
                over_pts = [int(2**3/i) for i in p]
                inputs = [(pts,over_pts[i],p[i]) for i in range(len(p))]
                mp.starmap(response_curve,inputs)
        # make animation from figures saved in response_curve
        fig = plt.figure()
        ims = []
        for i in p:
            im = plt.imshow(plt.imread('response_curve_changing_p/response_curve_{}.png'.format(i)),animated=True)
            plt.title('Response Curve for Pinhole Size = {}'.format(i))
            plt.xticks([])
            plt.yticks([])
            ims.append([im])
        ani = animation.ArtistAnimation(fig,ims,interval=1000,blit=True,repeat_delay=10)
        ani.save('response_curve_changing_p/response_curve.gif')
        plt.show()


    if False:
        cnms = np.zeros(16,dtype=np.float32)
        cnms[4] = 0.1
        with Pool(10) as p:
            pup,samp,rms = p.starmap(RMSE_one_config,[([2**i],[2**j],cnms) for i in range(3,9) for j in range(1,6)])

        plt.figure()
        plt.scatter(pup,rms,s=10,c=samp)
        plt.colorbar(label='Focal Plane Oversampling Factor')
        plt.xlabel('Pupil Width [pixels]')
        plt.ylabel('RMS Error')
        plt.show()
