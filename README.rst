=====
sedop
=====

sedop is a Monte-Carlo minimization code designed to optimally construct spectral energy distributions (SEDs) 
for sources of ultraviolet and X-ray radiation employed in numerical simulations of reionization and 
radiative feedback.  See `Mirocha et al. 2012 <http://arxiv.org/abs/1204.1944>`_ for details on the algorithm.

Getting started
---------------
To clone a copy and install: ::

    hg clone https://bitbucket.org/mirochaj/sedop sedop
    cd sedop
    python setup.py install

Currently, sedop depends on h5py, mpi4py, numpy, matplotlib, and scipy. mpi4py is not necessary if you only run in serial, but I would recommend building it as multi-frequency, multi-species optimizations can be rather expensive.  

Example
-------
A quick example can be run in an interactive Python session. The

::
	
	import sedop
	
	opt = sedop.OptimizeSED(source_type=1, num_bins=1)
	results = opt.run(prefix='test_bb', nsteps=1e4, ntrials=5, clobber=0,
	    restart=1)
		
This will create a file called 'results.h5' in your current working directory.  Since
TrackWalks = 1, you will also get a file called 'guesses_procN.h5' where N = 0 up to
however many processors you ran on.


Documentation
-------------
For more examples, checkout the documentation. For now, you'll have to build it locally, i.e., navigate to the ``doc`` folder and type ``make html``. Then, open ``_build/html/index.html``.

