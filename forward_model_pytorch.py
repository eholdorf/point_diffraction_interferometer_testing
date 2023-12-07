import numpy as np
import aotools
import torch
import os

def gernerate_phase_maps(max_zerns,pup_width):
    """
    Generate a set of phase maps for a given number of Zernike modes.

    Args:
    max_zerns (int): The maximum number of Zernike modes to consider.
    pup_width (int): Width of the pupil plane.

    Returns:
    numpy.ndarray: Array of Zernike mode coefficients.
    """
    # define the phase for up to max_zerns zernike modes
    zernikes = aotools.zernikeArray(max_zerns, pup_width, norm='rms')
    np.save('zernike_maps/{}_zernikes_{}.npy'.format(max_zerns,pup_width),zernikes)

def pinhole_mask(pinhole_size,fp_oversamp,pup_width):
    """
    Generate a pinhole mask.

    Args:
    pinhole_size (float): Size of the pinhole mask.
    fp_oversamp (int): Oversampling factor for the focal plane.
    pup_width (int): Width of the pupil plane.

    Returns:
    numpy.ndarray: The pinhole mask.
    """
    # define the pinhole mask
    mask = aotools.circle(fp_oversamp*pinhole_size, fp_oversamp*pup_width)
    np.save('zernike_maps/{}_pinhole_map_{}_{}.npy'.format(pinhole_size,pup_width,fp_oversamp),mask)

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
    #cnms = torch.from_numpy(cnms)
    # load the zernike modes if the file exists, else make it
    if os.path.isfile('zernike_maps/{}_zernikes_{}.npy'.format(max_zerns,pup_width)):
        zernikes = np.load('zernike_maps/{}_zernikes_{}.npy'.format(max_zerns,pup_width))
    else:
        gernerate_phase_maps(max_zerns,pup_width)
        zernikes = np.load('zernike_maps/{}_zernikes_{}.npy'.format(max_zerns,pup_width))
    zernikes = torch.from_numpy(zernikes)
    # the pupil function is just piston
    pup = zernikes[0]
    # load the pinhole mask if the file exists, else make it
    if os.path.isfile('zernike_maps/{}_pinhole_map_{}_{}.npy'.format(pinhole_size,pup_width,fp_oversamp)):
        mask = np.load('zernike_maps/{}_pinhole_map_{}_{}.npy'.format(pinhole_size,pup_width,fp_oversamp))
    else:
        pinhole_mask(pinhole_size,fp_oversamp,pup_width)
        mask = np.load('zernike_maps/{}_pinhole_map_{}_{}.npy'.format(pinhole_size,pup_width,fp_oversamp))
    mask = torch.from_numpy(mask)
    # mask amplitude modulation function, (half of input light plus half
    # of masked light). fftshift to sort the quadrants properly for fft-ing
    mod_func = torch.fft.fftshift((mask+1)*0.5)
    # build up the wavefront as a sum of the zernike modes with given amplitudes cnms
    # check type of cnms, if not a tensor make it one
    if not torch.is_tensor(cnms):
        cnms = torch.from_numpy(cnms)
    phi = torch.einsum("ijk,i->jk",zernikes,cnms)
    # pupil-plane complex amplitude is pupil amplitude with a phase delay:
    psi = pup * torch.exp(1j*2*np.pi/wavelength*(phi))
    # focal plane complex amplitude is FFT of pupil plane, modulated by
    # the mask modulation function (includes interference of two waves)
    fp_amp = torch.fft.fft2(psi,s=[pup_width*fp_oversamp]*2)*mod_func
    # sensor-plane intensity is inverse FFT of FP amplitude, cropped and squared
    intensity = torch.abs(torch.fft.ifft2(fp_amp)[:pup_width,:pup_width])**2
    # scale the intensity to match propogation.py
    return intensity/torch.max(intensity)

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
