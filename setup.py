#!/usr/bin/env python

from distutils.core import setup
from pyautoplot import __version__

setup(name='pyautoplot',
      version=__version__,
      description='Library to build interferometric data inspection tools',
      author='Michiel Brentjens',
      author_email='brentjens@astron.nl',
      url='',
      packages=['pyautoplot'],
      scripts=['scripts/msplots', 'scripts/tscount',
               'scripts/launch-msplots.sh',
               'scripts/launch-tscount.sh',
               'scripts/report_global_status',
               'scripts/create_html'],
     )
