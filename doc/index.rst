.. sedop documentation master file, created by
   sphinx-quickstart on Sat Jun 27 11:37:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
sedop
=====
sedop is a Monte-Carlo minimization code designed to optimally construct spectral energy distributions (SEDs)  for sources of ultraviolet and X-ray radiation employed in numerical simulations of reionization andradiative feedback.  See `Mirocha et al. 2012 <http://arxiv.org/abs/1204.1944>`_ for details on the algorithm.

Getting started
---------------
To clone a copy and install: ::

    hg clone https://bitbucket.org/mirochaj/sedop sedop
    cd sedop
    python setup.py install

Currently, sedop depends on h5py, mpi4py, numpy, matplotlib, and scipy. mpi4py is not necessary if you only run in serial, but I would recommend building it as multi-frequency, multi-species optimizations can be rather expensive.  

Example
-------
A quick example can be run in an interactive Python session. The following code snippet will run 5 independent Monte Carlo calculations, each with 10,000 steps, to find the optimal monochromatic representation of a :math:`10^5` K blackbody spectrum:

::
	
	import sedop
	
	opt = sedop.OptimizeSED(source_type=1, source_temperature=1e5, num_bins=1)
	results = opt.run(prefix='test_bb', nsteps=1e4, ntrials=5, clobber=0)
		
The `results` dictionary contains the end-points of each Monte Carlo run in energy (``Ef``) and normalization (``Ff``), as well as the starting points (``Ei`` and ``Fi``, respectively). The full trajectory of each random walk is saved in the ``Nsteps`` and ``Fsteps`` elements. A file called 'test_bb.hdf5' will also be saved in your current working directory containing the same contents as the ``results`` dictionary.


Documentation
-------------
For more examples, checkout the documentation. For now, you'll have to build it locally, i.e., navigate to the ``doc`` folder and type ``make html``. Then, open ``_build/html/index.html``.


Contents
--------
.. toctree::
    :maxdepth: 1

    Home <self>
    example_uv
    example_xray
    example_user
	parameters
	
	
