import numpy as np
import forward_model
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

def phase_retrieval(intensity, max_zerns, pup_width, fp_oversamp, pinhole_size, wavelength, cnms_init=None):
    """
    Perform phase retrieval on the given intensity distribution.

    Args:
    intensity (numpy.ndarray): The intensity distribution in the sensor plane.
    max_zerns (int): The maximum number of Zernike modes to consider.
    pup_width (int): Width of the pupil plane.
    fp_oversamp (int): Oversampling factor for the focal plane.
    pinhole_size (float): Size of the pinhole mask.
    wavelength (float): Wavelength of the light source in micron.
    cnms_init (numpy.ndarray): Initial guess for the Zernike mode coefficients.

    Returns:
    numpy.ndarray: The Zernike mode coefficients.
    """
    # define the objective function
    def objective(cnms):
        # calculate the forward model
        intensity_model = forward_model.forward_prop(max_zerns,cnms,pup_width,fp_oversamp,pinhole_size,wavelength)
        # calculate the error
        error = np.nansum((intensity_model-intensity)**2/intensity**2)
        error = 1-structural_similarity(intensity,intensity_model)
        return error
    # define the initial guess
    if cnms_init is None:
        cnms_init = np.zeros(max_zerns,dtype=np.float64)+1e-3
    # run the minimization
    res = least_squares(objective, cnms_init,xtol=1e-14,ftol=1e-14,max_nfev = 20000)
    # return the result
    return res.x

# test the above function
if __name__ == "__main__":
    # define the parameters
    max_zerns = 10
    cnms = np.random.randn(max_zerns)# np.zeros(max_zerns,dtype=np.float64) 
    cnms[4] = 0.1
    cnms[6] = -0.2
    pup_width = 2**6
    fp_oversamp = 2**2
    pinhole_size = 1
    wavelength = 0.589
    # run the forward model
    intensity = forward_model.forward_prop(max_zerns,cnms,pup_width,fp_oversamp,pinhole_size,wavelength)
    # run the phase retrieval
    cnms_retrieved = phase_retrieval(intensity, max_zerns, pup_width, fp_oversamp, pinhole_size, wavelength)
    # plot the result
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.imshow(intensity)
    
    plt.figure()
    plt.imshow(forward_model.forward_prop(max_zerns,cnms_retrieved,pup_width,fp_oversamp,pinhole_size,wavelength))
    
    plt.figure()
    plt.plot(np.arange(1,max_zerns+1,1),cnms,'o',label='input')
    plt.plot(np.arange(1,max_zerns+1,1),cnms_retrieved,'o',label='retrieved')
    plt.xlabel("Zernike Mode")
    plt.ylabel("Zernike Amplitude")
    plt.legend()
    plt.show()
