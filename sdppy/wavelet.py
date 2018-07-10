from scipy import stats
from scipy.special import gamma
from math import factorial
import numpy as np

chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

def chisqrpdf(v, df):
    """
    The function computes the probability P that, in a chi-square
    distribution with Df degrees of freedom, a random variable X is less than
    or equal to a user-specified cutoff value V.

    Parameters
    ----------
    v : list or tuple
    df : int
        The degrees of freedom.

    Returns
    -------
    return_value : ndarray
        The array of probability.
    """
    return 1 - chisqprob(v, df)


def chisqrcvf(p, df):
    """
    The function computes the cutoff value V in a chi-square
    distribution with df degrees of freedom such that the probability that a
    random variable X is greater than V is equal to a user-supplied
    probability P.

    Parameters
    ----------
    p : list or tuple
    df : int
        The degrees of freedom.

    Returns
    -------
    return_value :

    """
    if p < 0 or p > 1:
        print("p must be in the interval [0.0, 1.0]")
    if p == 0:
        return 1.0e12
    if p == 1:
        return 0.0
    if df < 0:
        print("Degrees of freedom must be positive.")

    if df == 1:
        up = 300.0
    else:
        if df == 2:
            up = 100.0
        else:
            if df > 2 and df <= 5:
                up = 30.0
            else:
                if df > 5 and df <= 14:
                    up = 20.0
                else:
                    up = 12.0
    below = 0
    while chisqrpdf(up, df) < (1 - p):
        below = up
        up = 2 * up
    return bisectpdf([1 - p, df], up, below)


def bisectpdf(a, up, below):
    """
    The function computes the cutoff value a such that the probabilty
    of an observation from the given distribution, less than x, is a(0).
    u and l are the upper and lower limits for x, respectively.

    Parameters
    ----------
    a : list or tuple
    u : int
        The upper limit for x.
    l : int
        The lower limit for x.

    Returns
    -------
    return_value :
    """
    z = None
    dl = 1.0e-6
    p = a[0]
    if (p < 0 or p > 1):
        return -1
    up = up
    low = below
    mid = below + (up - below) * p
    count = 1
    while (abs(up - low) > dl * mid) and (count < 100):
        if z is not None:
            if z > p:
                up = mid
            else:
                low = mid
        mid = (up + low) / 2.
        z = chisqrpdf(mid, a[1])
        count = count + 1
    return mid


def acovariance(x, lag):
    """
    The function returns the autocovariance.

    Parameters
    ----------
    x   : list or tuple or ndarray

    Lag : list or tuple or ndarray
        An n-element integer vector in the interval [-(n-2), (n-2)],
        specifying the signed distances between indexed elements of X.

    Returns
    -------
    return_value: ndarray
    """
    xa = np.average(x)
    sm = []
    for i in lag:
        s = 0
        for j in range(len(x) - abs(i)):
            s = s + (x[j] - xa) * (x[j + abs(i)] - xa)
        sm.append(s)
    sm = np.array(sm)
    return sm / len(x)


def acorrelation(x, lag):
    """
    The function returns the autocorrelation.

    Parameters
    ----------
    x : list

    lag : list
        An n-element integer vector in the interval [-(n-2), (n-2)],
        specifying the signed distances between indexed elements of X.

    Returns
    -------
    return_value : ndarray
    """
    xa = np.average(x)
    s = 0
    for i in range(len(x)):
        s = s + (x[i] - xa) ** 2
    acv = acovariance(x, lag) * len(x)
    return acv / s


def complexv(x, y):
    """
    The function returns complex scalars or arrays from two scalars or arrays.

    Parameters
    ----------
    x : list or tuple or ndarray
        The input array of real parts.
    y : list or tuple or ndarray
        The input array of imaginary parts.

    Returns
    -------
    return_value : ndarray
        The array of complex values.

    Examples
    --------
    >>> complexv([1,2,3,4],[1,1,3,3])
    array([ 1.+1.j,  2.+1.j,  3.+3.j,  4.+3.j])
    >>> complexv([[1,1],[2,2]],[[3,3],[0,0]])
    array([[ 1.+3.j,  1.+3.j],
       [ 2.+0.j,  2.+0.j]])
    """

    if np.size(x) != np.size(y):
        print("Dimension of first and second array not equals.")
    else:
        if np.size(x) == 1:
            return np.complex(x, y)
        else:
            return np.vectorize(np.complex)(x[:], y[:])


def rebin(a, new_shape):
    """
    The function resizes an array by averaging or repeating elements,
    new dimensions must be integral factors of original dimensions.

    Parameters
    ----------
    a : list or numpy.ndarray
        The input array.
    new_shape : (int, int)
        The new shape.

    Returns
    -------
    return_value : numpy array with new shape.

    Examples
    --------
    >>> x = [1,2,3,4,5,6]
    >>> rebin(x,(2,))
    array([ 2.,  5.])
    >>> rebin(x,(12,))
    array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    >>> x = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
    >>> rebin(x,(8,8))
    array([[1, 1, 2, 2, 3, 3, 4, 4],
       [1, 1, 2, 2, 3, 3, 4, 4],
       [1, 1, 2, 2, 3, 3, 4, 4],
       [1, 1, 2, 2, 3, 3, 4, 4],
       [1, 1, 2, 2, 3, 3, 4, 4],
       [1, 1, 2, 2, 3, 3, 4, 4],
       [1, 1, 2, 2, 3, 3, 4, 4],
       [1, 1, 2, 2, 3, 3, 4, 4]])
    >>>rebin(x,(2,2))
    array([[ 1.5,  3.5],
           [ 1.5,  3.5]])
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if len(a.shape) == 1:
        x = a.shape[0]
        new_x = new_shape[0]
        if new_x < x:
            return a.reshape((new_x, x / new_x)).mean(1)
        else:
            return np.repeat(a, new_x / x)
    else:
        x, y = a.shape
        new_x, new_y = new_shape
        if new_x < x:
            return a.reshape((new_x, x / new_x,
                              new_y, y / new_y)).mean(3).mean(1)
        else:
            return np.repeat(np.repeat(a, new_x / x,
                                       axis=0), new_y / y, axis=1)


def Morlet(k, scale, k0=-1):
    """
    Morlet wavelet transform.

    Parameters
    ----------
    scale : ndarray
        The vector of scale indices, given by S0*2^(j*DJ), j=0...J
        where J+1 is the total # of scales.
    k : ndarray
        1-d time series.

    Returns
    -------
    coi : ndarray
        if specified, then return the Cone-of-Influence, which is a
        vector of N points that contains the maximum period of useful
        information at that particular time.
    dofmin : int
        The degrees of freedom.
    Cdelta : float
        The reconstruction factor.
    period : ndarray
        The vector of "Fourier" periods (in time units) that
        corresponds to the SCALEs.
    """
    if k0 == -1:
        # wavenumber. For 'Morlet' default 6.0.
        k0 = 6.0
    n = np.size(k)
    expnt = -(scale * k - k0) ** 2. / 2. * (k > 0.)
    dt = 2. * np.pi / (n * k[1])
    # [Eqn(7)]
    # total energy=N
    norm = np.sqrt(2. * np.pi * (scale / dt)) * (np.pi ** (-0.25))
    # expnt[np.where(expnt < -100)] = -100.
    morlet = norm * np.exp(expnt)
    # morlet = morlet * (expnt > -100)#avoid underflow errors
    # Heaviside step function (Morlet is complex)
    morlet = morlet * (k > 0.)
    fourier_factor = (4. * np.pi) / (k0 + np.sqrt(2. + k0 ** 2))
    period = scale * fourier_factor
    # Cone-of-influence [Sec.3g]
    coi = fourier_factor / np.sqrt(2)
    # Degrees of freedom with no smoothing
    dofmin = 2
    Cdelta = -1
    if (k0 == 6):
        # reconstruction factor
        Cdelta = 0.776
    psi0 = np.pi ** (-0.25)
    return morlet, coi, dofmin, Cdelta, period, psi0


def Dog(k, scale, k0=-1):
    """
    DOG (derivative of Gaussian) wavelet transform.

    Parameters
    ----------
    scale : ndarray
        The vector of scale indices, given by S0*2^(j*DJ), j=0...J
        where J+1 is the total # of scales.
    k : ndarray
        The 1-d time series.

    Returns
    -------
    coi : ndarray
        If specified, then return the Cone-of-Influence, which is a
        vector of N points that contains the maximum period of useful
        information at that particular time.
    dofmin : int
        The degrees of freedom.
    Cdelta : float
        The reconstruction factor.
    period : ndarray
        The vector of "Fourier" periods (in time units) that
        corresponds to the SCALEs.
    """
    if k0 == -1:
        k0 = 2
    n = len(k)
    expnt = -(scale * k) ** 2 / 2
    dt = 2 * np.pi / (n * k[1])
    norm = np.sqrt(2 * np.pi * scale / dt) * np.sqrt(1 / gamma(k0 + 0.5))
    I = complexv(0, 1)
    expnt[np.where(expnt < -100)] = -100
    gauss = -norm * (I ** k0) * (scale * k) ** k0 * np.exp(expnt)
    fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * k0 + 1))
    period = scale * fourier_factor
    coi = fourier_factor / np.sqrt(2)
    # Degrees of freedom with no smoothing
    dofmin = 1
    Cdelta = -1
    psi0 = -1
    if k0 == 2:
        # reconstruction factor
        Cdelta = 3.541
        psi0 = 0.867325
    if k0 == 6:
        # reconstruction factor
        Cdelta = 1.966
        psi0 = 0.88406
    return gauss, coi, dofmin, Cdelta, period, psi0


def Paul(k, scale, k0=-1):
    """
    Paul wavelet transform.

    Parameters
    ----------
    scale : ndarray
        The vector of scale indices, given by S0*2^(j*DJ), j=0...J
        where J+1 is the total # of scales.
    k : ndarray
        The 1-d time series.

    Returns
    -------
    coi : ndarray
        If specified, then return the Cone-of-Influence, which is a vector
        of N points that contains the maximum period of useful information
        at that particular time.
    dofmin : int
        The degrees of freedom.
    Cdelta : float
        The reconstruction factor.
    period : ndarray
        The vector of "Fourier" periods (in time units) that
        corresponds to the SCALEs.
    """
    if k0 == -1:
        k0 = 4
    n = len(k)
    expnt = -(scale * k) * (k > 0)
    dt = 2 * np.pi / (n * k[1])
    norm = np.sqrt(2 * np.pi * (scale / dt) * ((2 ** k0) /
                                               np.sqrt(k0 *
                                                       factorial(2 * k0 - 1))))
    expnt[np.where(expnt < -100)] = -100
    paul = norm * ((scale * k) ** k0) * np.exp(expnt)
    paul = paul * (k > 0.)
    fourier_factor = 4 * np.pi / (2 * k0 + 1)
    period = scale * fourier_factor
    coi = fourier_factor * np.sqrt(2)
    dofmin = 2  # Degrees of freedom with no smoothing
    Cdelta = -1
    if (k0 == 4):
        # reconstruction factor
        Cdelta = 1.132
    psi0 = 2. ** k0 * factorial(k0) / np.sqrt(np.pi *
                                              factorial(2 * k0))
    return paul, coi, dofmin, Cdelta, period, psi0


def wave_coherency(time1, time2, scale1, scale2, wave1, wave2,
                   coi1, coi2, dt, dj=0.125, nosmooth=False,
                   verbose=False):
    """
    The function compute the wavelet coherency between two time series.

    Parameters
    ----------
    wave1 : ndarray
        The wavelet power spectrum for time series #1.
    time1 : ndarray
        The vector of times for time series #1.
    scale1 : ndarray
        The vector of scales for time series #1.
    wave2 : ndarray
        The wavelet power spectrum for time series #2.
    time2 : ndarray
        The vector of times for time series #2.
    scale2 : ndarray
        The vector of scales for time series #2.
    dt : float
        Amount of time between each Y value, i.e. the sampling time. If not
        input, then calculated from time1(1)-time1(0).
    dj : float
        The spacing between discrete scales. If not input, then calculated
        from scale1.
    verbose : bool
        If True, then print out the scales and system time.
    nosmooth : float
        If True, then just compute the global_coher, global_phase, and
        the unsmoothed cross_wavelet and return.

    Returns
    -------
    wave_coher : ndarray
        The wavelet coherency, as a function of time_out and scale_out.
    time_out : ndarray
        The time vector, given by the overlap of time1 and time2.
    scale_out : ndarray
        The scale vector of scale indices, given by the overlap of
        scale1 and scale2.
    coi_out : ndarray
        The vector of the cone-of-influence.
    global_coher : ndarray
        The global (or mean) coherence averaged over all times.
    global_phase : ndarray
        The global (or mean) phase averaged over all times.
    cross_wavelet : ndarray
        The cross wavelet between the time series.
    power1 : ndarray
        The wavelet power spectrum; should be the same as wave1 if time1
        and time2 are identical, otherwise it is only the overlapping  portion.
    nosmooth : bool
        Is set, then this is unsmoothed, otherwise it is smoothed.
    """
    # find overlapping times
    time_start = np.max([np.min(time1), np.min(time2)])
    time_end = np.min([np.max(time1), np.max(time2)])
    time1_start = np.min(np.where(time1 >= time_start))
    time1_end = np.max(np.where(time1 <= time_end))
    time2_start = np.min(np.where(time2 >= time_start))
    time2_end = np.max(np.where(time2 <= time_end))
    # find overlapping scales
    scale_start = np.max([np.min(scale1), np.min(scale2)])
    scale_end = np.min([np.max(scale1), np.max(scale2)])
    scale1_start = np.min(np.where((scale1 >= scale_start)))
    scale1_end = np.max(np.where((scale1 <= scale_end)))
    scale2_start = np.min(np.where((scale2 >= scale_start)))
    scale2_end = np.max(np.where((scale2 <= scale_end)))

    # cross wavelet & individual wavelet power
    cross_wavelet = (wave1[time1_start:time1_end, scale1_start:scale1_end] *
                     np.conj(wave2[time2_start:time2_end,
                                   scale2_start:scale2_end]))
    power1 = np.abs(wave1[time1_start:time1_end, scale1_start:scale1_end]) ** 2
    power2 = np.abs(wave2[time2_start:time2_end, scale2_start:scale2_end]) ** 2

    dt = time1[1] - time1[0]
    ntime = time1_end - time1_start + 1
    nj = scale1_end - scale1_start + 1
    dj = np.log(scale1[1] / scale1[0]) / np.log(2)
    scale = scale1[scale1_start:scale1_end]
    if verbose:
        print(dt, ntime, dj, nj)
    time_out = time1[time1_start:time1_end]
    scale_out = scale1[scale1_start:scale1_end]
    if len(coi1) == len(time1):
        coi_out = coi1[time1_start:time1_end]

    # calculate global coherency before doing local smoothing
    global1 = np.sum(power1, 1)
    global2 = np.sum(power2, 1)
    global_cross = np.sum(cross_wavelet, 1)
    global_coher = np.abs(global_cross) ** 2 / (global1 * global2)
    global_phase = 180. / np.pi * np.arctan2(global_cross.imag,
                                             global_cross.real)
    if nosmooth:
        return {'time_out': time_out,
                'scale_out': scale_out,
                'coi_out': coi_out,
                'global_coher': global_coher,
                'global_phase': global_phase,
                'cross_wavelet': cross_wavelet,
                'power1': power1,
                'power2': power2}
    # time-smoothing
    for i in range(0, nj - 1):
        nt = 4 * scale[i] / dt / 2 * 4 + 1
        time_wavelet = (np.arange(nt) - nt / 2) * dt / scale[i]
        # Morlet
        wave_function = np.exp(-time_wavelet ** 2 / 2.)
        # normalize
        wave_function = wave_function / np.sum(wave_function)
        nz = nt / 2
        zeros = np.arange(nz, dtype='complex')
        cross_wave_slice = np.concatenate([zeros, cross_wavelet[:, i], zeros])
        cross_wave_slice = np.convolve(cross_wave_slice, wave_function)
        cross_wavelet[:, i] = cross_wave_slice[nz:ntime + nz - 1]
        power_slice = np.concatenate([zeros, power1[:, i], zeros])
        power_slice = np.convolve(power_slice, wave_function)
        power1[:, i] = power_slice[nz:ntime + nz - 1]
        power_slice = np.concatenate([zeros, power2[:, i], zeros])
        power_slice = np.convolve(power_slice, wave_function)
        power2[:, i] = power_slice[nz:ntime + nz - 1]
        if verbose:
            print(i, scale[i])
        # normalize by scale
        # FIXME
    scales = rebin(scale, ntime, nj)
    cross_wavelet = cross_wavelet / scales
    power1 = power1 / scales
    power2 = power2 / scales
    # closest (smaller) odd integer
    nweights = np.fix(0.6 / dj / 2 + 0.5) * 2 - 1
    weights = np.tile(1., nweights)
    # normalize
    weights = weights / np.sum(weights)
    # scale-smoothing
    for i in range(0, ntime - 1):
        cross_wavelet[i, :] = np.convolve((cross_wavelet[i, :])[:], weights)
        power1[i, :] = np.convolve((power1[i, :])[:], weights)
        power2[i, :] = np.convolve((power2[i, :])[:], weights)
    # scale-smoothing
    wave_phase = 180. / np.pi * np.arctan(cross_wavelet.imag, cross_wavelet)
    wave_coher = (np.abs(cross_wavelet) ** 2) / (power1 * power2 > 1E-9)
    # wave_phase = wave_phase + 360.*(wave_phase LT 0.)
    return {'wave_coher': wave_coher,
            'wave_phase': wave_phase,
            'time_out': time_out,
            'scale_out': scale_out,
            'coi_out': coi_out,
            'global_coher': global_coher,
            'global_phase': global_phase,
            'cross_wavelet': cross_wavelet,
            'power1': power1,
            'power2': power2}


def wave_signif(y, dt, scale, dof=2, sigtest=1, core='morlet',
                param=-1, lag1=[0.0], siglvl=0.95, gws=None,
                confidence=False, psi0=None, cdelta=None):
    """
    The function compute the significance levels for a wavelet transform.

    Parameters
    ----------

    y : ndarray
        The time series, or, the variance of the time series.
    dt : float
        Amount of time between each y value, i.e. the sampling time.
    scale : ndarray
        The vector of scale indices, from previous call to WAVELET.
    sigtest : int
        0, 1, or 2. if omitted, then assume 0.
        if 0 (the default), then just do a regular chi-square test, i.e. Eqn
        (18) from Torrence & Compo.
        if 1, then do a "time-average" test, i.e. Eqn (23). In this case, DOF
        should be set to NA, the number of local wavelet spectra that were
        averaged together. For the Global Wavelet Spectrum, this would be NA=N,
        where N is the number of points in your time series.
        if 2, then do a "scale-average" test, i.e. Eqns (25)-(28). In this
        case, DOF should be set to a two-element vector [S1,S2], which gives
        the scale range that was averaged together. e.g. if one scale-averaged
        scales between 2 and 8, then DOF=[2,8].
    core : str
        A string giving the mother wavelet to use. Currently, 'morlet',
        'paul','dog' (derivative of Gaussian) are available. Default is
        'morlet'.
        param : optional mother wavelet parameter.
        For 'Morlet' this is k0 (wavenumber), default is 6.
        For 'Paul' this is m (order), default is 4.
        For 'DOG' this is m (m-th derivative), default is 2.
    lag1 : float
        The lag 1 Autocorrelation, used for SIGNIF levels. Default is 0.0
    siglvl : float
        The significance level to use. Default is 0.95
    dof : int
        Degrees-of-freedom for signif test.
        if sigtest=0, then (automatically) DOF = 2 (or 1 for MOTHER='DOG')
        if sigtest=1, then dof = NA, the number of times averaged together.
        if sigtest=2, then dof = [s1,s2], the range of scales averaged.
        if SIGTEST=1, then DOF can be a vector (same length as SCALEs), in
        which case NA is assumed to vary with SCALE. This allows one to average
        different numbers of times together at different scales, or to take
        into account things like the Cone of Influence. See discussion
        following Eqn (23) in Torrence & Compo.
    gws : ndarray
        The global wavelet spectrum. if input then this is used as the
        theoretical background spectrum, rather than white or red noise.
    confidence : bool
        If True, then return a Confidence interval.
        For sigtest=0,2 this will be two numbers, the lower & upper.
        For sigtest=1, this will return an array (J+1)x2, where J+1 is the
        number of scales.

    Returns
    -------

    signif : ndarray
        The significance levels as a function of SCALE,
        or if confidence = True, then confidence intervals

    period : ndarray
        The vector of "Fourier" periods (in time units) that corresponds
        to the scaless.

    fft_theor : ndarray
        The output theoretical red-noise spectrum as fn of period.
    """
    if len(y) == 1:
        varianc = y
    else:
        varianc = np.var(y)
    # check keywords & optional inputs
    if not (core.lower() in ("morlet", "dog", "paul")):
        print("Core must be 'morlet', 'paul' or 'dog'!")
        return -1
    if not isinstance(lag1, list):
        lag1 = [lag1]
    lag1 = lag1[0]
    J = len(scale) - 1
    # s0 = min(scale)
    dj = np.log(scale[1] / scale[0]) / np.log(2)
    m = 0
    if core.lower() == 'morlet':
        if param == -1:
            k0 = 6
        else:
            k0 = param
        # [Sec.3h]
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        empir = [2., -1, -1, -1]
        if k0 == 6:
            empir[1:] = [0.776, 2.32, 0.60]
    if core.lower() == 'paul':
        if param == -1:
            m = 4
        else:
            m = param
        fourier_factor = 4 * np.pi / (2 * m + 1)
        empir = [2., -1, -1, -1]
        if m == 4:
            empir[1:] = [1.132, 1.17, 1.5]
    if core.lower() == 'dog':
        if param == -1:
            m = 2
        else:
            m = param
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
        empir = [1., -1, -1, -1]
        if m == 2:
            empir[1:] = [3.541, 1.43, 1.4]
    period = scale * fourier_factor
    # Degrees of freedom with no smoothing
    dofmin = empir[0]
    # reconstruction factor
    cdelta = empir[1]
    # time-decorrelation factor
    gamma = empir[2]
    # scale-decorrelation factor
    dj0 = empir[3]
    # significance levels [Sec.4]
    # normalized frequency
    freq = dt / period
    # [Eqn(16)]
    fft_theor = (1 - lag1 ** 2) / (1 - 2 * lag1 *
                                   np.cos(freq * 2 * np.pi) + lag1 ** 2)
    # include time-series variance
    fft_theor = varianc * fft_theor
    if gws is not None:
        if len(gws) == (J + 1):
            fft_theor = gws
    signif = fft_theor
    if m == 6:
        empir[1:] = [1.966, 1.37, 0.97]
    # no smoothing, DOF=dofmin
    if sigtest == 1:
        dof = dofmin
        # [Eqn(18)]
        signif = fft_theor * chisqrcvf(1. - siglvl, dof) / dof
        if confidence:
            sig = (1. - siglvl) / 2.
            chisqr = dof / np.array([chisqrcvf(sig, dof),
                                     chisqrcvf(1. - sig, dof)])
            # chisqr
            signif = fft_theor
    # time-averaged, DOFs depend upon scale [Sec.5a]
    if sigtest == 2:
        if len(dof) < 1:
            dof = dofmin
        if (gamma == -1):
            print('Gamma (decorrelation factor) not defined for ' + core)
        if (len(dof) == 1):
            dof = np.arange(J + 1, dtype='f4') + dof
        dof = dof[np.where(dof > 1)]
        # [Eqn(23)]
        dof = dofmin * np.sqrt(1 + (dof * dt / gamma / scale[:len(dof)]) ** 2)
        # minimum DOF is dofmin
        dof = dof[np.where(dof > dofmin)]
        if not confidence:
            for a1 in range(0, J - 1):
                chisqr = chisqrcvf(1. - siglvl, dof[a1]) / dof[a1]
                signif[a1] = fft_theor[a1] * chisqr
        else:
            signif = np.arange((J + 1) * 2, dtype='f4').reshape(J + 1, 2)
            sig = (1. - siglvl) / 2.
            for a1 in range(0, J):
                chisqr = dof[a1] / [chisqrcvf(sig, dof[a1]),
                                    chisqrcvf(1. - sig, dof[a1])]
                signif[a1, :] = fft_theor[a1] * chisqr
    if sigtest == 3:
        if len(dof) != 2:
            print('DOF must be set to [S1,S2], the range of scale-averages')
        if cdelta == -1:
            print('Cdelta & dj0 not defined for ' + core)
        s1 = dof[0]
        s2 = dof[1]
        avg = scale[np.where((scale >= s1 and scale <= s2))]
        navg = len(avg)
        if navg < 1:
            print('No valid scales between ' + str(s1) + ' and ' + str(s2))
        s1 = min(scale[avg])
        s2 = max(scale[avg])
        # [Eqn(25)]
        savg = 1. / sum(1. / scale[avg])
        # power-of-two midpoint
        smid = np.exp((np.log(s1) + np.log(s2)) / 2.)
        # [Eqn(28)]
        dof = (dofmin * navg * savg / smid) * np.sqrt(1 +
                                                      (navg * dj / dj0) ** 2)
        # [Eqn(27)]
        fft_theor = savg * np.sum(fft_theor(avg) / scale(avg))
        chisqr = chisqrcvf(1. - siglvl, dof) / dof
        if confidence:
            sig = (1. - siglvl) / 2.
            chisqr = dof / [chisqrcvf(sig, dof), chisqrcvf(1. - sig, dof)]
        # [Eqn(26)]
        signif = (dj * dt / cdelta / savg) * fft_theor * chisqr
    return signif, period, fft_theor


def wavelet(y1, dt=0.25, lag1=0.0, s0=None, dj=0.125, param=6,
            siglvl=0.95, j=None, fft_theor=None, voice=None,
            verbose=False, dodaughter=False, nowave=False, recon=True,
            pad=True, _oct=None, core="morlet"):
    """
    Compute the wavelet transform of a 1D time series.

    Parameters
    ----------
    y1 :numpy.array, list
        the time series of length n.
    dt : float
        amount of time between each y value, i.e. the sampling time.
    lag1 : float
        lag 1 Autocorrelation, used for signif levels. Default is 0.0.
    s0 : float
        the smallest scale of the wavelet.  Default is 1.0*dt.
    dj : float
        the spacing between discrete scales. Default is 0.125. A smaller will
        give better scale resolution, but be slower to plot.
    param : int
        optional mother wavelet parameter.
        For 'Morlet' this is k0 (wavenumber), default is 6.
        For 'Paul' this is m (order), default is 4. Might be!
        For 'DOG' this is m (m-th derivative), default is 2.
    siglvl : float
        significance level to use. Default is 0.95.
    j : int
        the # of scales minus one. Scales range from s0 up to s0*2^(j*dj), to
        give a total of (j+1) scales. Default is j = (log2(n dt/s0))/dj.
    fft_theor : numpy.ndarray
        theoretical background spectrum as a function of Fourier
        frequency. This will be smoothed by the wavelet function and
        returned as a function of period.
    voice : int
        Voices in each octave. Default is 8. Higher gives better scale
        resolution,but is slower to plot.
    verbose : bool
        if set, then print out info for each analyzed scale.
    daughter : bool
        if initially set to True, then return the daughter wavelets.This is
        a complex array of the same size as wavelet. At each scale the
        daughter wavelet is located in the center of the array.
    no_wave : bool
        off calculation wave.
    recon : bool
        if set, then reconstruct the time series, and store in y. Note that
        this will destroy the original time series, so be sure to input a
        dummy copy of y.
    pad : bool
        if set, then pad the time series with enough zeroes to get n up to
        the next higher power of 2. This prevents wraparound from the end of
        the time series to the beginning, and also speeds up the FFT's used
        to do the wavelet transform. This will not eliminate all edge effects.
    _oct : float
        the # of octaves to analyze over. Largest scale will be s0*2^oct_.
        Default is (log2(n) - 1).

    Returns
    -------
    wave : numpy.ndarray
        the wavelet transform of y.
    coi : numpy.ndarray
        if specified, then return the Cone-of-Influence, which is  a vector
        of n points that contains the maximum period  of useful information
        at that particular time.
    ypad : numpy.ndarray
        returns the padded time series that was actually used in the wavelet
        transform.
    dof : numpy.ndarray
        degrees of freedom.
    y1 : numpy.ndarray
        reconstruction of the original data.
    period : numpy.ndarray
        the vector of "Fourier" periods (in time units) that corresponds
        to the scales.
    signif : numpy.ndarray
        output significance levels as a function of period.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array(range(0, 314 * 100, 1)) * 0.005
    >>> x1 = np.sin(t) + np.sin(t / 8) + np.sin(t / 16)
    >>> wv = wavelet(x1, dt=0.25, core='morlet')
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(1)
    >>> plt.imshow(wv.lpwr)
    >>> plt.axis('tight')
    >>> plt.show()
    """
    # Initial core of wavelet
    if core.lower() == "morlet":
        cwave = Morlet
    else:
        if core.lower() == "paul":
            cwave = Paul
        else:
            if core.lower() == "dog":
                cwave = Dog
    # ---
    n = len(y1)
    n1 = n
    # power of 2 nearest to N
    base2 = np.fix(np.log(n) / np.log(2) + 0.4999).astype(int)
    # ....check keywords & optional inputs
    if s0 is None:
        s0 = 2.0 * dt
    if voice is not None:
        dj = 1. / voice
    if dj is None:
        dj = 1. / 8
    if ((_oct is not None) and (np.size(_oct) == 1)):
        j = _oct / dj
    if j is None:
        j = np.fix((np.log(n * dt / s0) / np.log(2)) / dj)
    # remove mean
    ypad = y1 - np.mean(y1)
    # pad with extra zeroes, up to power of 2
    if pad:
        ypad = np.append(ypad, np.zeros(2 ** (base2 + 1) - n, dtype='f4'))
        n = len(ypad)
    # construct SCALE array & empty PERIOD & WAVE arrays
    # of scales
    na = j + 1
    # array of scales  2^j   [Eqn(9)]
    scale = 2. ** (np.arange(na) * dj) * s0
    # empty period array (filled in below)
    period = np.arange(na, dtype='f8')
    wave = np.arange(n * int(round(na)), dtype='complex').reshape(int(round(na)), n)
    # wavelet array
    if dodaughter:
        # empty daughter array
        daughter = wave
    # construct wavenumber array used in transform [Eqn(5)]
    k = (np.arange(n / 2.) + 1) * (2 * np.pi) / (n * dt)
    k = np.append(k, -k[0:n // 2 - 1][::-1])
    k = np.append(0, k)
    # compute FFT of the (padded) time series
    yfft = np.fft.ifft(ypad)

    if verbose:
        s = 'points = {0} \n s0 = {1} \n dj = {2} \n j = {3}'
        print(s.format(n1, s0, dj, np.fix(j)))
        if n1 != n:
            print('padded with {0} zeroes'.format(n - n1))
    if np.size(fft_theor) == n:
        fft_theor_k = fft_theor
    else:
        fft_theor_k = (1 - lag1 ** 2) / (1 - 2 * lag1 * np.cos(k * dt) +
                                         lag1 ** 2)
    fft_theor = np.zeros(int(na), dtype='f4')
    # loop thru each SCALE
    # scale
    for i in range(0, int(na), 1):
        psi_fft, coi, dofmin, cdelta, period1, psi0 = cwave(k, scale[i])
        if not nowave:
            # wavelet transform[Eqn(4)]
            wave[i, :] = np.fft.fft(yfft * psi_fft)
        # save period
        period[i] = period1
        fft_theor[i] = np.sum((np.abs(psi_fft) ** 2) * fft_theor_k) / n
        if dodaughter:
            # save daughter
            daughter[i, :] = np.fft.fft(psi_fft)
        if verbose:
            s = 'Verbose: \n i = {0} \n scale({0}) = {1} '
            s = s + "\n period({0}) = {2}"
            print(s.format(i, scale[i], period[i]))
    coi = coi * np.append(np.arange(n1 / 2, dtype='f4'),
                          np.arange(n1 / 2, dtype='f4')[::-1]) * dt
    # shift so DAUGHTERs are in middle of array
    if dodaughter:
        daughter = [daughter[n - n1 / 2:, :], daughter[0:n1 / 2 - 1, :]]
    # significance levels [Sec.4]
    sdev = np.var(y1)
    # include time-series variance
    fft_theor = sdev * fft_theor
    dof = dofmin
    # [Eqn(18)]
    signif = fft_theor * chisqrcvf(1. - siglvl, dof) / dof
    if recon:
        if cdelta == -1:
            y1 = -1
            print('Cdelta undefined, cannot reconstruct with this wavelet')
        else:
            dottem = np.dot(1. / np.sqrt(scale), wave.real)
            y1 = dj * np.sqrt(dt) / (cdelta * psi0) * dottem
            y1 = y1[0:n1]
    return WaveletResult(**{'wave': wave[:, :n1],
                            'ypad': ypad[:n1],
                            'dof': dof,
                            'period': period,
                            'signif': signif,
                            'scale': scale,
                            'coi': coi,
                            'y1': y1,
                            'psi0': psi0,
                            'cdelta': cdelta})


class WaveletResult(object):

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.lpwr = (np.abs(np.array(self.wave))) ** 2
        self.gpwr = np.sum(self.lpwr, 1)

__all__ = ["wavelet"]
