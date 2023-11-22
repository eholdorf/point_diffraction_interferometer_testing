import numpy as np
import matplotlib.pyplot as plt
import general_formulas as gf
import propagation as prop
import scipy.optimize
from multiprocessing import Pool

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

amps = np.linspace(-0.5, 0.5, 100)
rms = []

def rms_calculation(amp):
    cnms = np.zeros(max_zerns)
    cnms[5]=amp

    measured_intensity = f([cnms,frac,pinhole_size,max_zerns,wavelength,pup_width,fp_oversamp,mode_type])

    def g(x):
        amp = x
        interferogram = prop.propagate(amp, frac, pinhole_size, max_zerns,wavelength,pup_width,fp_oversamp,mode_type)
        return np.nansum((measured_intensity - interferogram)**2)
    
    phase = scipy.optimize.minimize(g, np.zeros(max_zerns)).x

    return gf.calc_rms(phase,cnms) * 589

with Pool(10) as mp:
    rms = mp.map(rms_calculation, amps)

plt.scatter(amps, rms)
plt.show()