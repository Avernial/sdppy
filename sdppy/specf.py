import numpy as np


def hilbert_spec(data, time=None, dt=None):
    """
    The function 'hilbert_spec' estimates the Hilbert-Huang amplitude
    spectrum of an input matrix of time series (e.g. intrinsic mode functions).

    Parameters
    ----------
    data : ndarray
        Matrix of time series (e.g. intrinsic mode functions), of type
        floating point.  Dimensions are length of time series by number of time
        series.
    dt : float
        The time step of the data in Data.  Of type floating point.  If time
        is input, Dt is determined from TIME.

    Returns
    -------
    return_value : ndarray
        The time-frequency-amplitude matrix containing the Hilbert
        amplitude spectrum.  Dimensions are number of frequency components by
        length of time series minus one.
    """
    # Constants and Options
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    # Number of components in the Data matrix
    if len(data.shape) == 1:
        data = [data, ]
        data = np.array(data)
    nt = len(data[0, :])
    ncomp = len(data[:, 0])
    # absolute constants
    im = np.complex(0, 1)
    # A numerical fix term
    # epsilon = 0.00001
    # Determine time coordinates of spectral output
    if time is not None:
        t = time
        dt = t[1] - t[0]
    else:
        if dt is None:
            dt = 1.
        else:
            t = dt * np.arange(nt, dtype='f4') + dt / 2.
    # Calculate the Instanteous Spectrum
    # Define the Hilbert transform of data
    z = 0. * im * data
    for i in range(0, ncomp, 1):
        z[i, :] = data[i, :] + im * np.real(hilbert(data[i, :]))
    # Transform z to polar coordinates
    # Modulus of z
    magz = np.abs(z)
    # Phase of z
    angz = np.arctan(np.imag(z) / np.real(z))
    idf = np.where(np.real(z) < 0)
    nid = len(idf)
    # Output matrices
    # Amplitude spectrum
    if nid != 0:
        angz[idf] = angz[idf] + np.pi
    spec = np.zeros(nt * nt, dtype='f4').reshape(nt, nt)
    # Instantaneous frequency
    instfreq = np.arange(nt * ncomp, dtype='f4').reshape(ncomp, nt)
    # Instantaneous frequency index
    instfreqid = np.arange(nt * ncomp, dtype='i').reshape(ncomp, nt)
    for i in range(0, ncomp, 1):
        # Unwrap polar coordinate phase
        angz[i, :] = np.unwrap(angz[i, :])
        # Calculate instantaneous frequency
        # This is the derivative of the phase (and convert to positive
        # frequencies).
        tmp = np.array(np.hstack([angz[i, 1] - angz[i, 0],
                                  first_diff(angz[i, :],
                                             direct='forward')[0:nt - 1]]))
        instfreq[i, :] = tmp / (2. * np.pi)
        # Instantaneous frequency number
        instfreqid[i, :] = np.floor(np.abs(instfreq[i, :]) * nt * 2)

        # Calculate Hilbert amplitude spectrum.
        # This is the Hilbert transform amplitude corresponding to the
        # given time and frequency
        ctr = 0
        # Iterate through time
        for j in range(0, nt - 1, 1):
            # Copy the frequency number
            idf = instfreqid[i, j]
            # Determine if it is a legal frequency number
            if (idf >= 0) and (idf <= nt - 1):
                # Convert to the power spectrum
                spec[j, idf] = spec[j, idf] + magz[i, j] ** 2
            else:
                ctr = ctr + 1
        if ctr != 0:
            print('Warning:  ' + str(ctr))
            print(' invalid frequency values on time series ' + str(i))
    # Determine time coordinates of spectral output
    freq = np.arange(nt, dtype='f4') / nt * 0.5 / dt
    return spec, freq


def imfhspectrum(imf, ln):
    """
    The function calculate the Hilbert spectrum for imf and do the
    average frame.

    Parameters
    ----------
    imf : ndarray
        The result returned by the function emd
    ln : int
        The time series length.

    Returns
    -------
    return_value : ndarray
        The average frame of hilbert spectrum.

    """
    cnt = len(imf) // ln
    hh, freq = hilbert_spec(imf[0:ln])
    for i in range(1, cnt - 1, 1):
        temp, freq = hilbert_spec(imf[ln * i:ln * (i + 1)])
        hh = hh + temp
    return np.log1p(hh), freq


def hilbert(data, direction=1):
    """
    The function return a series that has all periodic terms
    shifted by 90 degrees.

    Parameters
    ----------
    data : ndarray
        A floating- or complex-valued vector containing any number
        of elements.

    direction : int
        A flag for rotation direction.  Set D to +1 for a positive rotation.
        Set D to -1 for a negative rotation. If D is not provided, a positive
        rotation results.

    Returns
    -------
    return_value : ndarray
        The Hilbert transform of the data vector, X.  The output is a
        complex-valued vector with the same size as the input vector.
    """
    y = np.array(np.fft.fft(data) / len(data))
    n = len(y)
    i = np.complex(0.0, 1.0)
    if direction == -1:
        i = i * direction
    n2 = n // 2 - 1
    tmp = y[1:n2 + 1] * i
    y[1:n2 + 1] = tmp
    n2 = n - n2
    tmp = y[n2:n] / i
    y[n2:n] = tmp
    # go back to time domain
    y = np.fft.ifft(y) * len(y)
    for i in range(0, len(y), 1):
        if y[i].imag < 1e-12:
            y[i] = np.complex(y[i].real, 0)
    return y


def first_diff(data, direct='backward'):
    """
    The function returns the first difference vector of the input.

    Parameters
    ----------
    data : ndarray
        A vector of type integer or floating point.
    direct : str
        backward -  Forces calculation of the backward difference.  This is
        the default.

        forward -  Forces calculation of the forward difference.  The default
        is the backward difference.

        centered -  Forces calculation of the centered difference.  The default
        is the backward difference.

    Returns
    -------
    return_value : ndarray
        The first difference of the input vector.
    """
    # Variables

    # Vector length
    n = len(data)
    # Output
    fdif = np.zeros(n, dtype='f4')
    if direct not in ('backward', 'forward', 'centered'):
        print('Options must be backward, forward or centered')
        return -1
    # Calculate First Difference
    # Backward difference
    if direct == 'backward':
        # Backward difference
        id1 = np.arange(n, dtype='i')
        # First value = first value - last value (in other words, a fix)
        id2 = np.hstack([n - 1, np.arange(n - 1, dtype='i')])
        fdif = data[id1] - data[id2]
    # Forward difference
    if direct == 'forward':
        # Forward difference
        # Last value = first value - last value (in other words, a fix)
        id1 = np.hstack([np.arange((n - 1), dtype='i') + 1, 0])
        id2 = np.arange(n, dtype='i')
        fdif = data[id1] - data[id2]
    # Centred difference
    if direct == 'centered':
        # First value = second value - first value (in other words, a fix)
        # Last value = last value - second-last value (in other words, a fix)
        id1 = np.hstack([np.arange((n - 1), dtype='i') + 1, 0])
        id2 = np.hstack([n - 1, np.arange(n - 1, dtype='i')])
        fdif = (data[id1] - data[id2]) / 2.
    return fdif

__all__ = ["hilbert_spec", "imfhspectrum", "hilbert", "first_diff"]
