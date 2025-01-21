"""Generate a restructured text document that describes built-in bandpasses
and save it to this module's docstring for the purpose of including in
sphinx documentation via the automodule directive."""

from sncosmo.bandpasses import _BANDPASSES, _BANDPASS_INTERPOLATORS

__all__ = []  # so that bandpass_table is not documented.

# string.ascii_letters in py3
ASCII_LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

bandpass_meta = _BANDPASSES.get_loaders_metadata()
table_delim = "  ".join([15 * '=', 120 * '=', 8 * '=', 20 * '=', 24 * '='])
table_colnames = ("{0:15}  {1:120}  {2:8}  {3:20}  {4:24}"
                  .format('Name', 'Description', 'Data URL',
                          'Retrieved', 'Reference'))
urlnums = {}


def bandpass_table(setname):
    """Return a string containing a rst table of bandpasses in the set."""

    lines = [table_delim,
             table_colnames,
             table_delim]
    allrefs = []

    for m in bandpass_meta:
        if (
                m['filterset'] != setname or
                # special case of ztf position-dependent bandpass
                (setname == 'ztf' and 'ZTF-II IN2P3 participation group' in m['description'])
        ):
            continue

        reflink = ''
        urllink = ''
        retrieved = ''

        if 'reference' in m:
            reflink = '[{0}]_'.format(m['reference'][0])
            if m['reference'] not in allrefs:
                allrefs.append(m['reference'])

        if 'dataurl' in m:
            dataurl = m['dataurl']
            if dataurl not in urlnums:
                if len(urlnums) == 0:
                    urlnums[dataurl] = 0
                else:
                    urlnums[dataurl] = max(urlnums.values()) + 1
            urllink = '`{0}`_'.format(ASCII_LETTERS[urlnums[dataurl]])

        if 'retrieved' in m:
            retrieved = m['retrieved']

        lines.append("{0!r:15}  {1:120}  {2:8}  {3:20}  {4:24}".format(
            m['name'], m['description'], urllink, retrieved, reflink))

    lines.append(table_delim)
    lines.append("")

    for refkey, ref in allrefs:
        lines.append(".. [{0}] {1}".format(refkey, ref))

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Build the module docstring

# Get the names of all filtersets
setnames = []
for m in bandpass_meta:
    setname = m['filterset']
    if setname not in setnames and setname not in ('hsc', 'megacam6'):
        setnames.append(setname)

# For each set of bandpasses, write a heading, the table, and a plot.
lines = []
for setname in setnames:
    lines.append("")
    lines.append(setname)
    lines.append(len(setname) * "-")
    lines.append("")
    lines.append(bandpass_table(setname))
    lines.append("""
.. plot::

   from bandpass_plot import plot_bandpass_set
   plot_bandpass_set({0!r})
""".format(setname))

# Bandpass interpolators
bandpass_interpolator_meta = _BANDPASS_INTERPOLATORS.get_loaders_metadata()
setnames = {'megacampsf'}
for setname in setnames:
    names = [m['name'] for m in bandpass_interpolator_meta
             if m['filterset'] == setname]
    lines.append("")
    lines.append(setname)
    lines.append(len(setname) * "-")
    lines.append("")
    lines.append("These are radially-variable bandpasses. To get a Bandpass at a given radius, use ``band = sncosmo.get_bandpass('megacampsf::g', 13.0)``")
    lines.append("")
    lines.append("""
.. plot::

   from bandpass_plot import plot_bandpass_interpolators
   plot_bandpass_interpolators({0!r})
""".format(names))

# General bandpass interpolators
setnames = {'ztf', 'megacam6', 'hsc'}
for setname in setnames:
    names = [m['name'] for m in bandpass_interpolator_meta
             if m['filterset'] == setname]
    lines.append("")
    lines.append(setname)
    lines.append(len(setname) * "-")
    lines.append("")
    lines.append("These are focalplane variable bandpasses. To get a Bandpass at a given (x, y, sensor_id) position on the focal plane, use ``band = sncosmo.get_bandpass('{0}', x=0, y=0, sensor_id=1)``. To get the averaged Bandpass, use ``band = sncosmo.get_bandpass('{0}')``".format(names[0]))
    lines.append("")
    lines.append("""
.. plot::

   from bandpass_plot import plot_general_bandpass_interpolators
   plot_general_bandpass_interpolators({0!r})
""".format(setname))



# URL links accumulated from all the tables.
for url, urlnum in urlnums.items():
    lines.append(".. _`{0}`: {1}".format(ASCII_LETTERS[urlnum], url))
lines.append("")

__doc__ = "\n".join(lines)
