# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['mwdust']

IRSA_BASE_URL = \
    'http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={:.5f}+{:.5f}'

def mwdust(ra, dec, source='irsa'):
    """Return Milky Way E(B-V) at given coordinates.

    Parameters
    ----------
    ra : float
    dec : float
    source : {'irsa'}, optional
        Default is 'irsa', which means to make a web query of the IRSA 
        Schlegel dust map calculator. No other sources are currently
        supported.

    Returns
    -------
    mwebv : float
        Milky Way E(B-V) at given coordinates.
    """

    import urllib
    from xml.dom.minidom import parse

    # Check coordinates
    if ra < 0. or ra > 360. or dec < -90. or dec > 90.:
        raise ValueError('coordinates out of range')

    if source == 'irsa':
        try:
            u = urllib.urlopen(IRSA_BASE_URL.format(ra, dec))
            if not u:
                raise ValueError('URL query returned false')
        except:
            print 'E(B-V) web query failed'
            raise

        dom = parse(u)
        u.close()

        try:
            EBVstr = dom.getElementsByTagName('meanValue')[0].childNodes[0].data
            result = float(EBVstr.strip().split()[0])
        except:
            print "E(B-V) query failed. Do you have internet access?"
            raise

        return result
