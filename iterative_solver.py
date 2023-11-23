#%%
import numpy as np
import matplotlib.pyplot as plt
import general_formulas as gf
import propagation as prop
import scipy.optimize
from multiprocessing import Pool
import multiprocessing
import tqdm

# define the parameters
wavelength = 0.589
pup_width = 2**6
pinhole_size = 0.685
fp_oversamp = int(2**3/pinhole_size)
frac = 0.5
max_zerns = 30
mode_type = 'Zernike'

def f(x):
    amp, frac, pinhole_size, max_zerns,wavelength,pup_width,fp_oversamp,mode_type = x
    interferogram = prop.propagate(amp, frac, pinhole_size, max_zerns,wavelength,pup_width,fp_oversamp,mode_type)
    return interferogram

rms = []
amps = np.linspace(-2, 2, 100)
def rms_calculation(z):
    amps = np.linspace(-2, 2, 100)
    rmss = []
    for amp in tqdm.tqdm(amps):
        cnms = np.zeros(max_zerns)
        cnms[z]=amp

        measured_intensity = f([cnms,frac,pinhole_size,max_zerns,wavelength,pup_width,fp_oversamp,mode_type])

        def g(x):
            amp = x
            interferogram = prop.propagate(amp, frac, pinhole_size, max_zerns,wavelength,pup_width,fp_oversamp,mode_type)
            return np.nansum((measured_intensity - interferogram)**2)
        
        phase = scipy.optimize.minimize(g, np.zeros(max_zerns)).x
        rmss.append(gf.calc_rms(phase,cnms) * 589)

    return rmss

with Pool(multiprocessing.cpu_count()//2) as mp:
    # calculate the rms with tqdm to show progress
    rms = mp.map(rms_calculation, list(range(max_zerns)))

cmap = plt.get_cmap("gist_rainbow")
plt.figure()
for i in range(max_zerns):
    plt.scatter(amps, rms[i], label='Zernike {}'.format(i), s=10, marker='.', color=cmap(float(i)/max_zerns))
plt.savefig('rms_iterative_solver.png')
plt.show()