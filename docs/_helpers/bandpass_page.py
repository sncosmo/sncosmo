"""Generate a restructured text document that describes built-in bandpasses
and save it to this module's docstring for the purpose of including in
sphinx documentation via the automodule directive."""

from astropy.extern import six
from sncosmo.bandpasses import _BANDPASSES

# string.ascii_letters in py3
ASCII_LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

bandpass_meta = _BANDPASSES.get_loaders_metadata()
table_delim = "  ".join([11 * '=', 80 * '=', 14 * '=', 8 * '=', 12 * '='])
table_colnames = ("{0:11}  {1:80}  {2:14}  {3:8}  {4:12}"
                  .format('Name', 'Description', 'Reference', 'Data URL',
                          'Retrieved'))
urlnums = {}


def bandpass_table(setname):
    """Return a string containing a rst table of bandpasses in the set."""

    lines = [table_delim,
             table_colnames,
             table_delim]
    allrefs = []

    for m in bandpass_meta:
        if m['filterset'] != setname:
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

        lines.append("{0!r:11}  {1:80}  {2:14}  {3:8}  {4:12}".format(
            m['name'], m['description'], reflink, urllink, retrieved))

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
    if setname not in setnames:
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

# URL links accumulated from all the tables.
for url, urlnum in urlnums.items():
    lines.append(".. _`{0}`: {1}".format(ASCII_LETTERS[urlnum], url))
lines.append("")

__doc__ = "\n".join(lines)
