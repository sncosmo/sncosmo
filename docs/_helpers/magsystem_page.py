"""Generate a restructured text document that describes built-in magsystems
and save it to this module's docstring for the purpose of including in
sphinx documentation via the automodule directive."""

import string

from astropy.extern import six
from sncosmo.magsystems import _MAGSYSTEMS


lines = ['',
         '  '.join([10*'=', 60*'=', 35*'=', 15*'=']),
         '{0:10}  {1:60}  {2:35}  {3:15}'
         .format('Name', 'Description', 'Subclass', 'Spectrum Source')]
lines.append(lines[1])

urlnums = {}
for m in _MAGSYSTEMS.get_loaders_metadata():

    urllink = ''
    description = ''

    if 'description' in m:
        description = m['description']

    if 'url' in m:
        url = m['url']
        if url not in urlnums:
            if len(urlnums) == 0:
                urlnums[url] = 0
            else:
                urlnums[url] = max(urlnums.values()) + 1
        urllink = '`{0}`_'.format(string.ascii_letters[urlnums[url]])

    lines.append("{0!r:10}  {1:60}  {2:35}  {3:15}"
                 .format(m['name'], description, m['subclass'], urllink))

lines.extend([lines[1], ''])
for url, urlnum in six.iteritems(urlnums):
    lines.append('.. _`{0}`: {1}'.format(string.ascii_letters[urlnum], url))
lines.append('')
__doc__ = '\n'.join(lines)
