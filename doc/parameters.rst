:orphan:

Control Parameters
------------------
The simulated annealing algorithm used by *sedop* is pretty minimalist: it only has two real free parameters:

- The annealing frequency, ``anneal_freq``, which sets how often the "temperature" is adjusted. If, e.g., ``anneal_freq=20``, the temperature is adjusted every 20 steps.
- The annealing temperature adjustment, ``anneal_gamma``. Recall that the annealing temperature controls our tolerance of bad steps -- reducing its value makes us less tolerant. In order to be less tolerant of bad steps, the annealing temperature is reduced every ``anneal_freq`` steps by :math:`T \rightarrow \gamma T`, where :math:`gamma` is controlled by the parameter ``anneal_gamma``. By default, it takes a value of 0.98.

See Appendix A of `Mirocha et al. 2012 <http://arxiv.org/abs/1204.1944>`_ for more details.

