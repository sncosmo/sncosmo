
.. raw:: html

    <style media="screen" type="text/css">
      h1 { display:none; }
      th { display:none; }
      table.docutils td { border-bottom:none;  width:30em; }
      div.bodywrapper { max-width:75em; }
      pre { overflow-x:hidden; }
      div.leftcolumn { display: table-cell; padding-right: 5px; }
      div.rightcolumn { display: table-cell; padding-left: 5px; }
      strong { font-size: 1.2em; }
    </style>

*******
SNCosmo
*******


.. image:: _static/sncosmo_banner_96.png
    :width: 409px
    :height: 96px

SNCosmo aims to make high-level supernova cosmology analysis as easy
as possible, while still being completely extensible. It is built on
NumPy, SciPy and AstroPy.

.. raw:: html

    <div style="width: 100%; display: table;">
        <div style="display: table-row;">
            <div class="leftcolumn">

**Simulation & synthetic photometry**

::

    >>> import sncosmo
    >>> model = sncosmo.get_model('salt2')
    >>> model.set(c=0.05, x1=0.5, mabs=-19.3, z=1.0, t0=55100.)
    >>> model.bandmag('sdssr', 'ab', times=[55075., 55100., 55140.])
    array([ 27.1180182 ,  25.68243714,  28.28456537])

*See more in* :doc:`models` *and* :doc:`builtins/models`


**Model spectra**

::

    >>> from matplotlib import plot as plt
    >>> wl, flux = model.disp(), model.flux(time=55110.)
    >>> plt.plot(wl, flux)

.. image:: _static/example_spectrum.png  
   :width: 350px
   :height: 204px

*See more in* :doc:`models`

.. raw:: html

    </div><div class="rightcolumn">

**Fitting Light Curves**

::

    >>> res = sncosmo.fit_model(model, data, ['x1','c','z','mabs','t0'],
    ...                         bounds={'z': (0.3, 0.7)})
    >>> res.params['x1'], res.errors['x1']
    (0.14702167554607398, 0.033596743599762925)

*See more in* :doc:`fitting`


**Quick plots**

::

    >>> model.set(**res.params)  # set parameters to best-fit values
    >>> sncosmo.plotlc(data, model)

.. image:: _static/example_lc.png
   :width: 400px
   :height: 300px


**Photometric Typing**

::

    >>> typer = sncosmo.PhotoTyper()

.. raw:: html

    </div></div></div>

.. toctree::
   :hidden:
   :maxdepth: 1

   overview
   install
   models
   bandpasses
   magsystems
   photometric_data
   plotting
   fitting
   typing
   registry
   reference
   development
