__all__ = ["emd", "extrema", "interp", "zero_cross"]

import numpy as np
from scipy.interpolate import interp1d
from sdppy.specf import hilbert_spec


def zero_cross(data, position=False):
    """
    The function returns the number of zero crossing in a given
    time series.

    Parameters
    ----------
    data : ndarray
        The input array
    position : bool
        If set True, the function return the number of zero crossing and
        their position.

    Returns
    -------
    return_value : list
        The number of zero crossing and position.

    Examples
    --------
    >>> zero_cross([3,2,1,-1,-2])
    1
    >>> zero_cross([3,2,-1,-2,3,4], position = True)
    (2, array([1, 3]))
    """
    if position:
        zc = np.where(np.sign(data[1:]) != np.sign(data[:-1]))[0]
        return len(zc), zc
    else:
        return len(np.where(np.sign(data[1:]) != np.sign(data[:-1]))[0])


def interp(pos, val, new_pos, kind='cubic'):
    """
    The function 'interp' interpolates the data.

    Parameters
    ----------
    pos : list|ndarray
        The input array. Values MUST be monotonically increasing.

    val : list|ndarray
        The vector of ordinate values corresponding to X.

    new_pos : list|ndarray
        The vector of abcissae values for which the ordinate is
        desired. The values of T MUST be monotonically increasing.
    sigma : The amount of "tension" that is applied to the curve. The
        default value is 1.0. If sigma is close to 0, (e.g., .01),
        then effectively there is a cubic spline fit. If sigma
        is large, (e.g., greater than 10), then the fit will be like
        a polynomial interpolation.

    Returns
    -------
    return_value : ndarray
        The interpolated data.
    """
    f = interp1d(pos, val, kind=kind)
    return f(new_pos)


def extrema(x, minmax=False, strict=False, with_end=False):
    """
    The function will index the extrema of a given array x.

    Parameters
    ----------
    x : tuple|ndarray
        The input array.
    minmax : bool
        If True, function will return a list of joint minima and maxima,
        If False, function will return a two lists minima and maxima.
    strict : bool
        If True, will not index changes to zero gradient.
    with_end : bool
        If True, always include x[0] and x[-1].

    Returns
    -------
    return_value : list or two lists (min and max) of index.
    """
    # This is the gradient
    dx = np.zeros(len(x))
    dx[1:] = np.diff(x)
    dx[0] = dx[1]
    dx = np.sign(dx)

    threshold = 0
    if strict:
        threshold = 1

    d2x = np.diff(dx)

    # Take care of the two ends
    if with_end:
        d2x[0] = 2
        d2x[-1] = 2
    ind_max = np.nonzero(d2x > threshold)[0]
    ind_min = np.nonzero(d2x < threshold)[0]
    # Sift out the list of extremas
    if minmax:
        return np.sort(np.array(ind_max)), np.sort(np.array(ind_min))
    else:
        return np.sort(np.hstack([np.array(ind_max), np.array(ind_min)]))


def emd(data, quek=False, shiftfactor=0.3, splinemean=False, zerocross=False,
        interp_kind='cubic', epsilon=0.0001, ncheckimf=3):
    """
    The function estimates the empirical mode decomposition of a given data
    vector.

    Parameters
    ----------
    data : list|ndarray
        A floating point vector containing the input data values.
    quek : bool
        If set, the procedure test for imfs by checking the size of the
        difference between successive rounds but with a modified comparison
        as adopted by Quek et alii (2003). The default is without the
        modification.
    shiftfactor : float
        A floating point factor to be used in comparing normalised
        squared differences between successive rounds when testing for IMFs.
        The default is 0.3.
    splinemean : bool
        If set, the procedure estimates the local mean by splining
        between the mean of the extrema. The default is to take the mean of the
        splines through the extrema.
    zerocross : bool
        If set, the procedure tests for IMFs by comparing the number of
        extrema and zero crossings.  The default is by checking the size of the
        difference between successive rounds.
    interp_kind : str|int
        Specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic, 'cubic' where 'slinear', 'quadratic'
        and 'cubic' refer to a spline interpolation of first, second or third
        order) or as an integer specifying the order of the spline
        interpolator to use.
    epsilon : float
        factor for dealing with numerical precision
    ncheckimf : int
        number of shifting iterations to ensure stable IMF upon IMF candidate
        detection.

    Returns
    -------
    return_value : A list containing the intrinsic mode functions  (IMFs). The
             dimensions are number of time steps by number of IMFs.

    Examples
    --------
    >>> t = array(range(0, 314 * 200, 1)) * 0.01
    >>> data = sin(t) + sin(t / 8) + sin(t / 16)
    >>> imf = emd(data)
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(1)
    >>> plt.plot(imf[0]) #Plot first imf.
    >>> plt.show()
    """
    # Constants and Options
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    imf = []

    # Initialise check variable for determining loop exit.
    # Check = 0 means that we have nothing yet.
    check = 0

    # Check = 1 means that we have an IMF.
    checkimfval = 1
    # Check = 2 means that we have the residual.
    checkresval = 2
    # Check = 3 means that we exit the program.
    checkexitval = 3
    # Length of the time series
    ndata = len(data)
    data_std = np.std(data)
    # Initialise the vector to be decomposed (it is altered with each step)
    x = data

    # Decompose the Input Vector into Its IMFs
    # Iterate until signal has been decomposed
    while check < checkexitval:
    # Check if we have extracted everything (ie if you have the residual).
    # Find local extrema for minimum and maximum envelopes.
        nextrema = len(extrema(x))
        # Check for at least 1 extremum.
        if nextrema <= 2:
            check = checkresval
        # Check for very small residual.
        if np.std(x) < epsilon * data_std:
            check = checkresval
        # Remember what x was
        x0 = x

        # Initialise checkimf variable for determining stable IMF
        checkimf = 0
        checkres = 0

        # Iterate while the IMF criterion is not yet reached.
        # These criteria are incorporated into the Check variable.
        while check == 0:
            # Find local extrema s1for minimum and maximum envelopes
            # temp = extrema( x, minima=minima, maxima=maxima, /flat )
            minima, maxima = extrema(x, minmax=True)

            # Add a constant extension to the ends of the maxima and minima
            # vectors.
            # This is to get a better spline fit at the ends.
            # This is done by adding two cycles of a wave of wavelength
            # 2*abs(maxima[0]-minima[0]) onto the beginning, and similarly for
            # the end.

            # Period of beginning wave
            if len(maxima) != 0 and len(minima != 0):
                period0 = 2 * np.abs(maxima[0] - minima[0])
                # Period of end wave
                period1 = 2 * np.abs(maxima[-1] - minima[-1])
                # Extend the extrema vectors
                maxpos = np.hstack([maxima[0] - 2 * period0,
                                    maxima[0] - period0,
                                    maxima,
                                    maxima[-1] + period1,
                                    maxima[-1] + 2 * period1])
                maxval = np.hstack([x[maxima[0]],
                                    x[maxima[0]],
                                    x[maxima],
                                    x[maxima[-1]],
                                    x[maxima[-1]]])
                minpos = np.hstack([minima[0] - 2 * period0,
                                    minima[0] - period0,
                                    minima,
                                    minima[-1] + period1,
                                    minima[-1] + 2 * period1])
                minval = np.hstack([x[minima[0]],
                                    x[minima[0]],
                                    x[minima],
                                    x[minima[-1]],
                                    x[minima[-1]]])
                # Estimate local mean.
                # If we want to take the spline of the means of the extrema
                if splinemean:
                    meanpos = [0]
                    meanval = [0]
                    # If the first extremum is a minimum, do it first
                    if minpos[0] < maxpos[0]:
                        meanpos = np.hstack([0,
                                             [(minpos[0] + maxpos[0]) / 2]])
                        meanval = np.hstack([0,
                                             [(minval[0] + maxval[0]) / 2.]])
                    # Now iterate through all maxima, taking the average of
                    # this maximum and the following minimum, and the following
                    # minimum and following maximum.
                    for i in range(0, len(maxima) + 4 - 1, 1):
                        # Determine the position of the next minimum after
                        # this maximum
                        tmp = np.where(minpos > maxpos[i])[0]
                        id1 = np.min(tmp)
                        nid = np.where(tmp == id1)[0]
                        # If such a minimum exists
                        if nid != 0:
                        # Add the average position and value to our collection
                            meanpos.extend([(maxpos[i] + minpos[id1]) / 2])
                            meanval.extend([(maxval[i] + minval[id1]) / 2.])
                            # Determine the position of the next maxmum after
                            # this minimum
                            # nid = len(tmp)
                            tmp = np.where(maxpos > minpos[id1])
                            id2 = np.min(tmp)
                            nid = np.where(tmp == id2)[0]
                            # If such a maximum exists
                            if nid != 0:
                            # Add the average position and value to our
                            # collection
                                meanpos.extend([(maxpos[id2] +
                                                 minpos[id1]) / 2])
                                meanval.extend([(maxval[id2] +
                                                 minval[id1]) / 2.])
                    # Measure the number of estimates we have
                    nmean = len(meanpos) - 1
                    # Sort the estimates (not guaranteed by our method) and
                    # remove initialising values
                    idf = np.sort(meanpos[1:nmean])
                    meanpos = np.array(meanpos)[1 + np.array(idf, dtype='int')]
                    meanval = np.array(meanval)[1 + np.array(idf, dtype='int')]
                    # Estimate the local mean through a spline interpolation
                    localmean = interp(meanpos, meanval, np.arange(ndata),
                                    kind=interp_kind)
                    # If we want to take the mean of the splines of the extrema
                else:
                    # Spline interpolate to get maximum and minimum envelopes
                    maxenv = interp(maxpos, maxval, np.arange(ndata),
                                    kind=interp_kind)
                    minenv = interp(minpos, minval, np.arange(ndata),
                                    kind=interp_kind)
                    # Estimate the local mean as the mean of these envelopes
                    localmean = (minenv + maxenv) / 2.

                # Substract local mean from current data
                xold = x
                x = x - localmean
                # If the IMF criterion is the extrema/zero crossings comparison
                if zerocross:
                    # Count the number of zero crossings
                    nzeroes = zero_cross(x)
                    # Count the number of extrema
                    nextrema = len(extrema(x))
                    # Check if the number of zero crossings equals the number
                    # of extrema, to within one.
                    if nextrema - nzeroes <= 1:
                        # Count this as a candidate IMF
                        checkimf = checkimf + 1
                    else:
                        # Do not count this as a candidate IMF
                        checkimf = 0
                # If the IMF criterion is checking the size of the difference
                # between successive rounds.
                if not(zerocross):
                # Measure which will be used to stop the sifting process.
                # Huang refers to this as the standard deviation (SD) even
                # though it is not.

                # Calculate SD the traditional way
                    if not(quek):
                        sd = np.sum(((xold - x) ** 2) / (xold ** 2 + epsilon))
                # Or Quek et alii's modified way
                    else:
                        sd = np.sum((xold - x) ** 2) / np.sum(xold ** 2)

                # Compare sd value against threshold
                    if sd < shiftfactor:
                # Count this as a candidate IMF
                        checkimf = checkimf + 1
                    else:
                # Do not count this as a candidate IMF
                        checkimf = 0
                if np.std(x) < epsilon * data_std:
                    checkres = checkres + 1

                # Check to see if we have a satisfied IMF
                if checkimf == ncheckimf:
                    check = checkimfval
                if checkres == ncheckimf:
                    check = checkimfval
            else:
                check = 1

        imf.append(x)

        if check == checkresval:
                check = checkexitval
        else:
            check = 0
        # Substract the extracted IMF type filter textfrom the signal
        x = np.array(x0) - np.array(x)
    return imf


def join_imf(imf):
    """
    The function returns 1d array of imf.

    Parameters
    ----------
    imf : list
        The list of imfs.

    Returns
    -------
    result : ndarray
    """
    result = []
    for i in imf:
        result.extend(i)
    return result


class EmdResult():
    """
    Class for wrapping emd result for easy access to the imf.
    The class provides an iterator over intrinsinc mode functions.
    """
    # : The intrinsic mode functions.
    imf = None
    # : The number of intrinsic function.
    size = None

    def __init__(self, imf):
        self.imf = imf
        self.size = len(self.imf)

    def get_imf(self):
        """
        Returns intrinsic mode functions.
        """
        return join_imf(self.imf)

    def __getitem__(self, num):
        return self.imf[num]

    def __iter__(self):
        for i in range(self.size):
            yield(self[i])

    def __len__(self):
        return self.size

    def get_hspectr(self):
        """
        Return Hilbert spectrum and frequency.
        """
        hh, freq = hilbert_spec(self[0])
        for i in range(1, int(self.size)):
            temp, freq = hilbert_spec(self[i])
            hh = hh + temp.copy()
        return np.log1p(hh), freq

    def get_recon(self):
        return np.sum(self.imf, 0)
