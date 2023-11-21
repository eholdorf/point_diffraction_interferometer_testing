import numpy as np
import aotools

def forward_prop(max_zerns,cnms, pup_width,fp_oversamp,pinhole_size,wavelength):
    """
    Perform forward propagation of light through a point diffraction 
    interferometer with phase aberrations.

    Args:
    max_zerns (int): The maximum number of Zernike modes to consider.
    cnms (numpy.ndarray): Array of Zernike mode coefficients.
    pup_width (int): Width of the pupil plane.
    fp_oversamp (int): Oversampling factor for the focal plane.
    pinhole_size (float): Size of the pinhole mask.
    wavelength (float): Wavelength of the light source in micron.

    Returns:
    numpy.ndarray: The intensity distribution in the sensor plane 
                   after forward propagation.
    """
    # define the phase for up to max_zerns zernike modes
    zernikes = aotools.zernikeArray(max_zerns, pup_width, norm='rms')
    # the pupil function is just piston
    pup = zernikes[0]
    # define the pinhole mask
    mask = aotools.circle(fp_oversamp*pinhole_size, fp_oversamp*pup_width)
    # mask amplitude modulation function, (half of input light plus half
    # of masked light). fftshift to sort the quadrants properly for fft-ing
    mod_func = np.fft.fftshift((mask+1)*0.5)
    # build up the wavefront as a sum of the zernike modes with given amplitudes cnms
    phi = np.einsum("ijk,i->jk",zernikes,cnms)
    # pupil-plane complex amplitude is pupil amplitude with a phase delay:
    psi = pup * np.exp(1j*2*np.pi/wavelength*(phi))
    # focal plane complex amplitude is FFT of pupil plane, modulated by
    # the mask modulation function (includes interference of two waves)
    fp_amp = np.fft.fft2(psi,s=[pup_width*fp_oversamp]*2)*mod_func
    # sensor-plane intensity is inverse FFT of FP amplitude, cropped and squared
    intensity = np.abs(np.fft.ifft2(fp_amp)[:pup_width,:pup_width])**2
    return intensity

# test the above function
if __name__ == "__main__":
    # define the parameters
    max_zerns = 20
    cnms = np.zeros(max_zerns,dtype=np.float64) #np.random.randn(max_zerns)
    cnms[18] = 1
    pup_width = 2**6
    fp_oversamp = 2**2
    pinhole_size = 1
    wavelength = 0.589
    # run the function
    intensity = forward_prop(max_zerns,cnms,pup_width,fp_oversamp,pinhole_size,wavelength)
    # plot the result
    import matplotlib.pyplot as plt
    plt.imshow(intensity)
    plt.show()
