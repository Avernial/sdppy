.. _contents:

Welcome to SDPpy's documentation!
=================================

sdppy is an open-source Python library for spectral data analysis.

The package contains ported EMD (Empirical Mode Decomposition) and wavelet
with language IDL.

Wavelet software was provided by C. Torrence and G. Compo, and is available at
http://paos.colorado.edu/research/wavelets/.

EMD software was provided by Daithi A. Stone (stoned@atm.ox.ac.uk).

Requirements
------------
* `Python 3+ <http://www.python.org/download/>`_
* `NumPy/SciPy <http://scipy.org/Download>`_

Installation
------------
* Download `sdppy <https://github.com/Avernial/sdppy/archive/master.zip>`_.
* Unpack sdppy-master.zip
* cd sdppy-master
* python3 setup.py build
* sudo python3 setup.py install



Documentation
-------------

..  toctree::

	reference/emd.rst
	reference/wavelet.rst
	reference/specf.rst

Examples
--------

..  toctree::


	reference/usage_emd.rst
	reference/usage_wavelet.rst
    
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

