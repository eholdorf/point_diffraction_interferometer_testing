import numpy as np
import matplotlib.pyplot as plt
import general_formulas as gf
import propagation as prop
import control_matrix as cm

# Define the parameters
wavelength = 0.589
num_pixels = 2**6# [2**i for i in range(4,10)]
oversamp = [int(2**i/0.685) for i in range(1,4)]
p = 0.685
frac = 0.5

plt.figure()
# generate the control matrix
colours = ['black','red','green','blue','orange']
print('Changing oversampling...')
for j,amp in enumerate([1e-3]):
    for num in oversamp:
        num = int(num/p)
        IM_inv = cm.generate_matrix(max_zerns=20,pup_width=num_pixels,
        fp_oversamp=num,pinhole_size=p,wavelength=wavelength,ratio=frac,amp=1e-4,mode='Zernike')

        # generate the interferogram 
        cnms = np.zeros(20)
        cnms[10] = 0.3
        
        intensity = prop.propagate(cnms,frac,p,max_zerns=20,pup_width=num_pixels,
                                fp_oversamp=num,wavelength=wavelength,
                                mode_type='Zernike')
        intensity_no_aberrations = prop.propagate(np.zeros(20),frac,p,max_zerns=20,pup_width=num_pixels,
                                fp_oversamp=num,wavelength=wavelength,
                                mode_type='Zernike')
        # retrieve the phase
        phase = np.dot(IM_inv,intensity.ravel() - intensity_no_aberrations.ravel())
        phase[0] = 0

        rms = gf.calc_rms(phase,cnms) * 589
        print(num,rms)
        plt.plot([num],[rms],'o',color=colours[j],label='Amp = {}'.format(amp))
plt.xlabel('Focal Plane Oversampling (pixels)')
plt.ylabel('RMS error (nm)')
plt.savefig('oversampling_error.png')

    
# Define the parameters
wavelength = 0.589
num_pixels = [2**i for i in range(3,12)]
oversamp = int(2**3/0.685)
p = 0.685
frac = 0.5

plt.figure()
print('Changing number of points...')
# generate the control matrix
colours = ['black','red','green','blue','orange']
for j,amp in enumerate([1e-3]):
    for num in num_pixels:
        IM_inv = cm.generate_matrix(max_zerns=20,pup_width=num,
        fp_oversamp=oversamp,pinhole_size=p,wavelength=wavelength,ratio=frac,amp=1e-4,mode='Zernike')
        
        # generate the interferogram 
        cnms = np.zeros(20)
        cnms[10] = 0.3
        intensity = prop.propagate(cnms,frac,p,max_zerns=20,pup_width=num,
                                fp_oversamp=oversamp,wavelength=wavelength,
                                mode_type='Zernike')
        
        intensity_no_aberrations = prop.propagate(np.zeros(20),frac,p,max_zerns=20,pup_width=num,
                                fp_oversamp=oversamp,wavelength=wavelength,
                                mode_type='Zernike')
        # retrieve the phase
        phase = np.dot(IM_inv,intensity.ravel() - intensity_no_aberrations.ravel())
        phase[0] = 0

        rms = gf.calc_rms(phase,cnms) * 589
        print(num,rms)
        plt.plot([num],[rms],'o',color=colours[j],label='Amp = {}'.format(amp))
plt.xlabel('Pupil Width (pixels)')
plt.ylabel('RMS error (nm)')
plt.savefig('pupil_width_error.png')
plt.show()