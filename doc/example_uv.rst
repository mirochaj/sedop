:orphan:

Example: Simple UV Spectrum
---------------------------
The default `source_type=1` is a blackbody. Expanding on the quick example from the landing page, but with 3 energy bins, we have

::

	import sedop
	import numpy as np
	import matplotlib.pyplot as pl
	
	opt = sedop.OptimizeSED(source_type=1, num_bins=3)
	results = opt.run(prefix='test_bb', nsteps=1e4, ntrials=5)

.. note :: To run more trials, simply set ``clobber=0`` and ``restart=1`` as 
	keyword arguments to the ``run`` call. New results will be appended to 
	pre-existing results automatically.


First, to simply print the end-points of each Monte Carlo run, do:

::

	for j, bin in enumerate(range(opt.pf['num_bins'])):
	    print(results['Ef'][j,:])
	    print(results['Ff'][j,:])
	
To plot the trajectories of individual random walks, you can do something like	
	
::

	fig1, ax1 = pl.subplots(1, 1, num=1)
	
	Ntrials = results['Esteps'].shape[1]
	
	colors = 'k', 'b', 'c', 'm'
	for i in range(Ntrials):  
	    for j in range(opt.pf['num_bins']):
	        ax1.plot(results['Esteps'][j,i,:], color=colors[j], alpha=0.1,
	            label='bin {}'.format(j+1) if i == 0 else None)
	
	ax1.set_xlabel('step #')
	ax1.set_ylabel(r'$E \ [\mathrm{eV}]$')
	ax1.legend()
	
and similarly for the SED normalization:	
	
::
	
	fig2, ax2 = pl.subplots(1, 1, num=2)
	
	for i in range(Ntrials):
	    for j in range(opt.pf['num_bins']):
	        ax2.plot(results['Fsteps'][j,i,:], color=colors[j], alpha=0.1,
	            label='bin {}'.format(j+1) if i == 0 else None)
	
	ax2.set_xlabel('step #')
	ax2.set_ylabel(r'$F$')
	ax2.legend()
	
	
There are more analysis routines available in the analysis module, whihc you can initialize either via file or dictionary, e.g.,

::

	anl = sedop.Analyze(prefix='test_bb')
	# Alternative: anl = sedop.Analyze(prefix=results, pf=opt.pf)
	
	
	
	


