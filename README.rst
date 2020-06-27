=====
sedop
=====

sedop is a Monte-Carlo minimization code designed to optimally construct spectral energy distributions (SEDs) 
for sources of ultraviolet and X-ray radiation employed in numerical simulations of reionization and 
radiative feedback.  The methods paper is `Mirocha et al. 2012 <http://arxiv.org/abs/1204.1944>`_.

Getting started
---------------
To clone a copy and install: ::

    hg clone https://bitbucket.org/mirochaj/sedop sedop
    cd sedop
    python setup.py install

Currently, sedop depends on h5py, mpi4py, numpy, matplotlib, and scipy. mpi4py is not necessary if you only run in serial, but I would 
recommend building it as multi-frequency, multi-species optimizations can be rather expensive.  

Example
-------
sedop can be run in serial, or in an embarrassingly parallel way.  To get started, let's open
up a parameter file (call it 'pf.dat'), and compute the optimal monochromatic SED for a 
10^5 K blackbody spectrum: ::

    SourceType                  = 1         # Blackbody
    SourceTemperature           = 1e5       
    SourceMinEnergy             = 13.6      
    SourceMaxEnergy             = 1e2
    
    MinimizationMethod          = 0         # Only consider tau = 0
    RateIntegral                = [0, 1]    # 0 = Phi, 1 = Psi (defined in paper)
    MultiSpecies                = 0         # Hydrogen only
    NumberOfBins                = 1         # Monochromatic
    NumberOfWalks               = 100       # Number of MC trials
    NumberOfSteps               = 1000      # Steps per trial
    TrackWalks                  = 1         # Store each step in each trial
    MaximumEnergyStep           = 5.        # Largest step allowed in eV
    MaximumNormalizationStep    = 0.05      # Largest step allowed in fraction of Lbol

To run the optimization, copy the sedop driver script (sedop/bin/SEDOP.py) to your current 
working directory and type: ::

    python SEDOP.py pf.dat
    
or: ::

    mpirun -np N python SEDOP.py pf.dat    

This will create a file called 'results.h5' in your current working directory.  Since
TrackWalks = 1, you will also get a file called 'guesses_procN.h5' where N = 0 up to
however many processors you ran on.

User-Defined SEDs
-----------------
I've recently implemented the parameter 'SpectrumFile' which can be used to 
override all source/spectrum parameters, and perform optimization on a spectrum
provided in an ASCII or HDF5 file.  If ASCII, the file should be two columns: first,
the source emission energies in eV, and second, the specific luminosity emitted in
each energy bin.  If HDF5, the file must contain at least two datasets, one named
'E' and the other named 'L_E', which contain the emission energies and specific
luminosities, respectively.  

Analyzing the Data
------------------

To do some simple analysis of the output, open up a python (or ipython) session and use 
built-in analysis routines, or look at the raw data itself:

>>>
>>> import sedop.analysis as sa
>>> 
>>> # Supply parameter file to initialize dataset ('ds') object
>>> ds = sa.Analyze('./pf.dat', load_walks = True) 
>>>
>>> # Have a quick look at the 'best' solution
>>> ds.best 
>>>
>>> # Show the PDF for the energy 
>>> ds.hist()
>>> 
>>> # Plot the evolution for 5 random walks.
>>> ds.walk_plot(Nrand = 5)
>>> 
>>> # Look at how our optimal SED compares to the numerically
>>> # computed value of Phi.
>>> ds.compare()
>>>

For more examples, check the doc/examples folder. 

The `current documentation <https://bitbucket.org/mirochaj/sedop/downloads/sedop_manual.pdf>`_ 
is available as a PDF.  More coming soon!
