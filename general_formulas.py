import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def calc_rms(observed_data,true_data):
    """
    Calculate the root mean square (RMS) error between observed data and true data.

    Parameters:
    - observed_data (numpy.ndarray): Array containing the observed data.
    - true_data (numpy.ndarray): Array containing the true (reference) data.

    Returns:
    float: The root mean square (RMS) error between observed and true data.

    Formula:
    RMS = sqrt(sum((observed_data - true_data)**2) / len(observed_data))

    The root mean square error is a measure of the average magnitude of the differences
    between corresponding elements of the observed and true data arrays.

    Example:
    observed_data = np.array([1, 2, 3, 4])
    true_data = np.array([0.9, 2.1, 3.2, 4.2])
    rms_error = calc_rms(observed_data, true_data)
    print("Root Mean Square Error:", rms_error)
    """
    return np.sqrt(np.sum((observed_data-true_data)**2)/len(observed_data))

def convert_rad_to_wavelength(error):
    """
    Converts an angular error in radians to its corresponding wavelength in a wave.

    Parameters:
    - error (float): The angular error in radians.

    Returns:
    - float: The equivalent error in units of wavelength.

    Formula:
    The conversion is based on the relationship between angular frequency (w) and wavelength (λ) in a wave.
    The formula used here is: wavelength = angular_error / (2 * pi)

    Example:
    >>> convert_rad_to_wavelength(6.283)  # Equivalent to 1 full revolution (2*pi radians)
    1.0
    """
    return error/(2*np.pi) 

def convert_rad_to_nm(error,wavelength):
    """
    Converts an angular error in radians to its corresponding distance in nanometers along a wave.

    Parameters:
    - error (float): The angular error in radians.
    - wavelength (float): The wavelength of the wave in nanometers.

    Returns:
    - float: The distance in nanometers corresponding to the given angular error along the wave.

    Formula:
    The conversion is based on the relationship between angular frequency (w) and wavelength (λ) in a wave.
    The formula used here is: distance = (angular_error / (2 * pi)) * wavelength

    Example:
    >>> convert_rad_to_nm(6.283, 500)  # Equivalent to 1 full revolution (2*pi radians) for a 500 nm wave
    500.0
    """
    return error/(2*np.pi)*wavelength

def linear_line(x,m,b):
    """
    Calculates the y values of a line with a given slope and y-intercept.
    """
    return m*x + b
def quadratic_line(x,a,b,c):
    """
    Calculates the y values of a quadratic with given coefficients.
    """
    return a*x**2 + b*x + c

def exponential_line(x,a,b):
    """
    Calculates the y values of an exponential with given coefficients.
    """
    return a*np.exp(b*x)

def calc_fit(x,y,func):
    """
    Calculates the best fit of a given function to a set of data.

    Parameters:
    - x (numpy.ndarray): Array containing the x values of the data.
    - y (numpy.ndarray): Array containing the y values of the data.
    - func (function): The function to fit to the data.

    Returns:
    - numpy.ndarray: Array containing the fit to the data and the best fit 
                     values of the function parameters.

    Example:
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([1, 2, 3, 4])
    >>> calc_fit(x, y, linear_line)
    array([1., 0.])
    """
    popt, pcov = curve_fit(func, x, y)
    return func(x,*popt),popt