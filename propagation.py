import numpy as np
import aotools

def propagate(amp,frac,pinhole_size,max_zerns = 16, wavelength=0.589,pup_width=2**6,fp_oversamp=2**2, mode_type = 'Zernike'):
    """
    Propagates light through a point diffraction interferometer.

    Args:
        amp (float): Amplitudes of the Zernike modes.
        frac (float): Fractional transmission of the pinhole.
        pinhole_size (int): Diameter of the pinhole as a ratio of 1.22lam/D.
        max_zerns (int, optional): Maximum number of Zernike modes to generate (default is 16).
        wavelength (float, optional): Wavelength of the light in micrometers (default is 0.589 μm).
        pup_width (int, optional): Width of the pupil grid in pixels (default is 2^6).
        fp_oversamp (int, optional): Oversampling factor for the focal plane (default is 2^2).
        mode_type (str, optional): Type of Zernike modes to generate, options are: 'Zernike' or 'KL' (default is 'Zernike').

    Returns:
        ndarray: Intensity distribution at the camera plane after propagation.

    """
    # generate the zernike functions
    # make a padded pupil so the focal plane smapled well
    padded_pupil = np.zeros((pup_width*fp_oversamp,pup_width*fp_oversamp),dtype=complex)
    if mode_type == 'Zernike':
        zerns = aotools.zernikeArray(max_zerns,pup_width,norm="rms")
    elif mode_type == 'KL':
        zerns = aotools.zernikeArray(max_zerns,pup_width,norm='rms')
        zerns[1:] = aotools.make_kl(max_zerns-1,pup_width,ri=1e-4)[0]
    else:
        raise ValueError("mode_type must be either 'Zernike' or 'KL'")

    # borrow piston mode as the pupil function
    pup = zerns[0] 

    phi = np.einsum("ijk,i->jk",zerns,amp)
    beam = pup * np.exp(1j*2*np.pi/wavelength*phi)

    mid_int = int(pup_width*fp_oversamp/2)
    padded_pupil[mid_int - int(pup_width/2):mid_int + int(pup_width/2),
                 mid_int - int(pup_width/2):mid_int + int(pup_width/2)] = beam

    # now need to propagate from the pupil plane to the focal plane
    focal = aotools.opticalpropagation.lensAgainst(padded_pupil, wavelength*1e-6, 1/(fp_oversamp*pup_width), 0.1)

    # make the pinhole mask
    mask = aotools.circle(fp_oversamp*pinhole_size,fp_oversamp*pup_width)
    mask = np.where(mask == 0,frac,mask)

    # pass the light through the pinhole
    focal *= mask

    # propagate to the camera plane
    camera = aotools.opticalpropagation.oneStepFresnel(focal,wavelength*1e-6,2.44*wavelength*1e-6/(pup_width),0.1)
    
    mid_int += 1
    camera = camera[mid_int - int(pup_width/2):mid_int + int(pup_width/2),
                 mid_int - int(pup_width/2):mid_int + int(pup_width/2)]

    return abs(camera)**2

# test the function
if __name__=="__main__":
    import matplotlib.pyplot as plt
    # define the parameters
    max_zerns = 16
    zern_current = np.arange(0,max_zerns,dtype=np.int32)
    amp = np.zeros(max_zerns,dtype=np.float64)
    pinhole_size = 0.695
    wavelength = 0.589
    frac=0.5
    # run the forward model
    intensity_no_abberations = propagate(amp,frac,pinhole_size,pup_width=2**9,fp_oversamp=2**5,mode_type='Zernike')

    test_two = propagate(amp,frac,pinhole_size,pup_width=2**9,fp_oversamp=2**6,mode_type='Zernike')

    plt.figure()
    plt.imshow(test_two)
    plt.colorbar()
    plt.figure()
    plt.imshow(intensity_no_abberations)
    plt.colorbar()

    plt.figure()
    plt.imshow(test_two - intensity_no_abberations)
    plt.colorbar()

    plt.show()