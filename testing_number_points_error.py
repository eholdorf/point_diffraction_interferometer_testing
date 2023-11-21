import numpy as np
import matplotlib.pyplot as plt
import general_formulas as gf
import propagation as prop
import control_matrix as cm

# Define the parameters
wavelength = 0.589
num_pixels = [2**i for i in range(4,10)]
oversamp = int(2**3/0.685)
p = 0.685
frac = 0.5

plt.figure()
# generate the control matrix
colours = ['black','red','green','blue','orange']
for j,amp in enumerate([1e-1]):#,1e-3,1e-2,1e-1,1]):
    for num in num_pixels:
        IM_inv = cm.generate_matrix(max_zerns=10,pup_width=num,
        fp_oversamp=oversamp,pinhole_size=p,wavelength=wavelength,ratio=frac,amp=1e-4,mode='Zernike')

        # generate the interferogram 
        cnms = np.zeros(10)
        cnms[5] = amp
        intensity = prop.propagate(cnms,frac,p,max_zerns=10,pup_width=num,
                                fp_oversamp=oversamp,wavelength=wavelength,
                                mode_type='Zernike')
        intensity_no_aberrations = prop.propagate(np.zeros(10),frac,p,max_zerns=10,pup_width=num,
                                fp_oversamp=oversamp,wavelength=wavelength,
                                mode_type='Zernike')
        # retrieve the phase
        phase = np.dot(IM_inv,intensity.ravel() - intensity_no_aberrations.ravel())
        phase[0] = 0

        rms = gf.calc_rms(phase,cnms) * 589
        plt.plot([num],[rms],'o',color=colours[j],label='Amp = {}'.format(amp))
plt.xlabel('Number of pixels')
plt.ylabel('RMS error (nm)')
#plt.legend()
plt.show()

    