#!/usr/bin/env python

from distutils.core import setup

setup(name='pyautoplot',
      version='0.3',
      description='Library to build interferometric data inspection tools',
      author='Michiel Brentjens',
      author_email='brentjens@astron.nl',
      url='',
      packages=['pyautoplot'],
      scripts=['scripts/msplots', 'scripts/tscount',
               'scripts/launch-msplots.sh',
               'scripts/launch-tscount.sh'],
     )
