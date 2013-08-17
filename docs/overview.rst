********
Overview
********

At the core of the library is a class (and subclasses) for
representing supernova models: A model is essentially a spectroscopic
time series that may vary as a function of one or more
parameters. Simulation, fitting and typing can all be naturally built
on this core functionality.

Key Features
------------

- **Built-ins:** There are many other built-in supernova models
  accessible by name (both Type Ia and core-collapse), such as Hsiao,
  Nugent, and models from PSNID. Common bandpasses and magnitude
  systems are also built-in and available by name (as demonstrated
  above).

- **Object-oriented and extensible:** New models, bandpasses, and
  magnitude systems can be defined, using an object-oriented interface.

- **Fast:** Fully NumPy-ified and profiled. Generating
  synthetic photometry for 100 observations spread between four
  bandpasses takes on the order of 1 millisecond (depends on model
  and bandpass sampling).

Stability
---------

The basic features of the models, bandpasses and magnitude systems can
be considered "fairly" stable.  The fitting and typing functionalities
are more experimental and the API may change as it gets more
real-world testing.

History
-------

.. toctree::
   :maxdepth: 1

   whatsnew/0.2
