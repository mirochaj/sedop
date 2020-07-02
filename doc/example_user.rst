:orphan:

Example: User-Defined Spectrum
------------------------------
If you don't want to use the built-in SED options, that's OK -- it's easy to swap in an arbitrary spectrum. There are two ways to do this:

- Provide a lookup table via arrays or from a file.
- Provide a Python function that computes an SED.

The basic approach will look very similar to the code in :doc:`example_uv`. The only thing that really changes is that one must provide ``OptimizeSED`` with the keyword arggument ``spectrum_file`` or ``spectrum_func``. 

For example, for a sanity check we can make a :math:`10^5` K blackbody spectrum and save it to a file (must be hdf5 or two-column plain-text for now):

::

	import h5py
	import sedop
	import numpy as np
	
	# Create a sedop RadiationSource instance (blackbody is default)
	pf = SetDefaultParameterValues()
	src = RadiationSource(pf)

	# Sample the H-ionizing spectrum
	Earr = np.arange(13., 100, 1.)
	Larr = src.Spectrum(Earr)

	# Save to an hdf5 file.
	with h5py.File('bbsed.hdf5', 'w') as f:
	    f.create_dataset('E', data=Earr)
	    f.create_dataset('L_E', data=Larr)

Now, we can run the optimization as before, and set ``spectrum_file``, e.g.,

::

	opt = sedop.OptimizeSED(source_type=1, num_bins=3, spectrum_file='pl_uvsed.hdf5')
	results = opt.run(prefix='test_bb_from_file', nsteps=1e4, ntrials=5, clobber=1, restart=0)
		
This should reproduce the results of :doc:`example_uv`, though not exactly given the random nature of the Monte Carlo algorithm.

.. note :: You can also provide the spectrum as arrays, e.g., with ``spectrum_file=(Earr, Larr)`` above.

Lastly, if you'd prefer to provide a Python function via ``spectrum_func``, just beware that what is really needed is a Python object with a ``Spectrum`` method. For example, you could do something like:

::

    class MySpectrum(object):
        def __init__(self, **parameters_i_need):
            pass
            # Set attributes
			
        def Spectrum(self, E):
            """
            Parameters
            ----------
            E : int, float, np.ndarray
            	Photon energy in eV.
            	
            """
            return (E / 1e2)**1.5 # for example
			

As long as your spectrum is defined with respect to photon *energy*, not *number*, this should work. In other words, your ``Spectrum`` function should return values proportional to the luminosity in units of :math:`\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{Hz}^{-1}` or :math:`\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{eV}^{-1}`, not :math:`\mathrm{photons} \ \mathrm{s}^{-1} \ \mathrm{Hz}^{-1}` or :math:`\mathrm{photons} \ \mathrm{s}^{-1} \ \mathrm{\unicode{x212B}}^{-1}`.

Then, run via

::

	opt = sedop.OptimizeSED(source_type=1, num_bins=3, spectrum_func=MySpectrum)
	results = opt.run(prefix='test_sed_from_func', nsteps=1e4, ntrials=5, clobber=1, restart=0)

