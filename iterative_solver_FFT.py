#%%
import numpy as np
import matplotlib.pyplot as plt
import general_formulas as gf
import forward_model as prop
import scipy.optimize
from multiprocessing import Pool
import multiprocessing
import tqdm

def f(x):
    amp, frac, pinhole_size, max_zerns,wavelength,pup_width,fp_oversamp,mode_type = x
    interferogram = prop.forward_prop(max_zerns,amp, pup_width,fp_oversamp,pinhole_size,wavelength)
    return interferogram


def rms_calculation(z):
    wavelength = 0.589
    pup_width = 2**9
    pinhole_size = 0.685
    fp_oversamp = int(2**7/pinhole_size)
    frac = 0.5
    max_zerns = 20
    mode_type = 'Zernike'
    amps = np.linspace(-2, 2, 11)
    rmss = []
    for amp in tqdm.tqdm(amps):
        cnms = np.zeros(max_zerns)
        cnms[z]=amp

        measured_intensity = f([cnms,frac,pinhole_size,max_zerns,wavelength,pup_width,fp_oversamp,mode_type])

        def g(x):
            amp = x
            interferogram = prop.forward_prop(max_zerns,amp, pup_width,fp_oversamp,pinhole_size,wavelength)
            return np.nansum((measured_intensity - interferogram)**2)
        
        phase = scipy.optimize.minimize(g, np.zeros(max_zerns),options={'disp':True}).x
        
        rmss.append(gf.calc_rms(phase,cnms) * 589)

    return rmss

def iterative_solver(measured_intensity,frac, pinhole_size, max_zerns,wavelength,pup_width,fp_oversamp,mode_type):

    def g(amp):
        interferogram = prop.forward_prop(max_zerns,amp, pup_width,fp_oversamp,pinhole_size,wavelength)
        return np.nansum((measured_intensity - interferogram)**2)
    
    phase = scipy.optimize.minimize(g, np.zeros(max_zerns),tol=1e-10,options={'disp':True}).x
    return phase

if __name__ == '__main__':
    rms = []
    amps = np.linspace(-0.1, 0.1, 11)
    wavelength = 0.589
    pup_width = 2**8
    pinhole_size = 0.685
    fp_oversamp = int(2**3/pinhole_size)
    frac = 0.5
    max_zerns = 20
    mode_type = 'Zernike'
    # generate the rms curves for varing Zernike amplitudes
    if False:
        with Pool(10) as mp:
            # calculate the rms with tqdm to show progress
            rms = mp.map(rms_calculation, list(range(max_zerns)))

        cmap = plt.get_cmap("turbo")
        plt.figure(figsize=(10,10))
        for i in range(max_zerns):
            plt.scatter(amps, rms[i], label='Zernike {}'.format(i), s=10, marker='o', color=cmap(float(i)/max_zerns))
        plt.legend()
        plt.xlabel('Zernike Amplitude')
        plt.ylabel('RMS (nm)')
        plt.savefig('rms_iterative_solver_labelled.png')
        plt.show()

    # look at how the number of points in the fit affects the rms
    if True:
        # define the parameters
        wavelength = 0.589
        pinhole_size = 0.685
        pup_width = 2**6
        fp_oversamp = int(2**3/pinhole_size)
        frac = 0.5
        max_zerns = 20
        mode_type = 'Zernike'
        print('Changing oversampling rate...')
        plt.figure(figsize=(10,10))
        for pup in tqdm.tqdm([2**i for i in range(1,8)]):
            cnms = np.zeros(max_zerns)
            cnms[10] = 0.3
            measured_intensity = prop.forward_prop(max_zerns,cnms, pup_width,int(pup/pinhole_size),pinhole_size,wavelength)
            phase  = iterative_solver(measured_intensity, frac, pinhole_size, max_zerns,wavelength,pup_width,int(pup/pinhole_size),mode_type)
            rms = gf.calc_rms(phase,cnms)*589
            plt.plot([pup],[rms], label='RMS', marker='o', color='black')

            plt.title('RMS vs Oversampling')
            plt.xlabel('Oversampling')
            plt.ylabel('RMS (nm)')
            plt.savefig('figures/oversampling_error_iterative_method.png')
        plt.close()
        print('Changing number of points...')
        plt.figure(figsize=(10,10))
        for pup in tqdm.tqdm([2**i for i in range(3,12)]):
            cnms = np.zeros(max_zerns)
            cnms[10] = 0.3
            intensity = prop.propagate(cnms, frac, pinhole_size, max_zerns,wavelength,pup,fp_oversamp,mode_type)
            phase  = iterative_solver(intensity, frac, pinhole_size, max_zerns,wavelength,pup,fp_oversamp,mode_type)
            rms = gf.calc_rms(phase,cnms)*589
            plt.plot([pup],[rms], label='RMS', marker='o', color='black')
        plt.title('RMS vs Pupil Size')
        plt.xlabel('Pupil Size (pixels)')
        plt.ylabel('RMS (nm)')
        plt.savefig('figures/pupil_width_error_iterative_method.png')
        plt.show()

    