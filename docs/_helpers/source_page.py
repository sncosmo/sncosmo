"""Generate a restructured text document that describes built-in sources
and save it to this module's docstring for the purpose of including in
sphinx documentation via the automodule directive."""

import string

from sncosmo.models import _SOURCES


lines = [
    '',
    '  '.join([20*'=', 7*'=', 10*'=', 27*'=', 30*'=', 7*'=', 20*'=']),
    '{0:20}  {1:7}  {2:10}  {3:27}  {4:30}  {5:7}  {6:50}'.format(
        'Name', 'Version', 'Type', 'Subclass', 'Reference', 'Website', 'Notes')
    ]
lines.append(lines[1])

urlnums = {}
allnotes = []
allrefs = []
for m in _SOURCES.get_loaders_metadata():

    reflink = ''
    urllink = ''
    notelink = ''

    if 'note' in m:
        if m['note'] not in allnotes:
            allnotes.append(m['note'])
        notenum = allnotes.index(m['note'])
        notelink = '[{0}]_'.format(notenum + 1)

    if 'reference' in m:
        reflink = '[{0}]_'.format(m['reference'][0])
        if m['reference'] not in allrefs:
            allrefs.append(m['reference'])

    if 'url' in m:
        url = m['url']
        if url not in urlnums:
            if len(urlnums) == 0:
                urlnums[url] = 0
            else:
                urlnums[url] = max(urlnums.values()) + 1
        urllink = '`{0}`_'.format(string.ascii_letters[urlnums[url]])

    lines.append("{0!r:20}  {1!r:7}  {2:10}  {3:27}  {4:30}  {5:7}  {6:50}"
                 .format(m['name'], m['version'], m['type'], m['subclass'],
                         reflink, urllink, notelink))

lines.extend([lines[1], ''])
for refkey, ref in allrefs:
    lines.append('.. [{0}] `{1}`__'.format(refkey, ref))
lines.append('')
for url, urlnum in urlnums.items():
    lines.append('.. _`{0}`: {1}'.format(string.ascii_letters[urlnum], url))
lines.append('')
for i, note in enumerate(allnotes):
    lines.append('.. [{0}] {1}'.format(i + 1, note))
lines.append('')
__doc__ = '\n'.join(lines)
