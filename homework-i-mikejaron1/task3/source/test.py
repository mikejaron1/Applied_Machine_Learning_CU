import numpy as np

def std_means(a, axis=0):
    """
    Compute the arithmetic mean and standard deviation along the specified axis.
    Returns the average and standard deviation of the array elements.  The average
    and standard deviation is taken over the flattened array by default, otherwise 
    over the specified axis. `float64` intermediate and return values are used for 
    integer inputs.
    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.

    Returns
    -------
    m : ndarray
        Returns a new array containing the mean values
    n : ndarray
        Returns a new array containing the standard deviation values
        
    """
    m = np.mean(a[axis])
    n = np.std(a[axis])

    return m,n 


	
