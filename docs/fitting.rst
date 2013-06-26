*******************
Light Curve Fitting
*******************

Suppose we have data in a file that looks like this, and we know the true
redshift is 0.5::
 
    time band flux fluxerr zp zpsys
    55070.0 sdssg -0.263064256628 0.651728140824 25.0 ab
    55072.0512821 sdssr -0.836688186816 0.651728140824 25.0 ab
    55074.1025641 sdssi -0.0104080573938 0.651728140824 25.0 ab
    55076.1538462 sdssz -0.0794771107707 0.651728140824 25.0 ab
    55078.2051282 sdssg 0.897840283912 0.651728140824 25.0 ab
    ...

Read the data from the file:

    >>> meta, data = sncosmo.readlc('mydatafile.dat')  # read the data

Get a model to fit, and set the redshift:

    >>> model = sncosmo.get_model('salt2')
    >>> model.set(z=0.5)


    >>> sncosmo.fit_model(model, data, ['c', 'mabs', 'x1', 't0'],
    ...                   params_start={'c':0.0,'mabs':-19., 'x1':0.}) # returns minimum chi^2
    49.454298041985474
    >>> model.params  # The model's parameters have now been set
    OrderedDict([('fscale', 1.1359644507654488e-17), ('m', 22.851972081015116),
                 ('mabs', -19.440540416846492), ('t0', 55100.001499062993),
                 ('z', 0.5), ('c', 0.219388164837543), ('x1', 1.0280567245583592)])
