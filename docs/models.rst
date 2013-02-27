**********************
Using Supernova Models
**********************

Models of astrophysical transient sources. A model is anything where
the spectrum as a function of phase can be parameterized by an
arbitrary number of parameters. For example, the Hsiao and Nugent SN
templates are zero parameter models (not counting amplitude): there is
a single spectrum for a given phase. The SALT2 model has two
parameters (:math:`x1` and :math:`c`) that determine a unique spectrum
as a function of phase. The Hsiao and Nugent models can be described
using a single class (`models.TimeSeries`) whereas the SALT2 model
needs a separate subclass (`models.SALT2`). All models derive from a
common abstract base class (`models.Transient`).

Examples
--------


Reference / API
---------------

