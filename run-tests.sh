#!/bin/bash

PYTHONPATH="`pwd`:$PYTHONPATH"
TESTOUTPUT=test-output

rm -rf $TESTOUTPUT
mkdir $TESTOUTPUT

python2 make_cluster_dirs.py

NOSETESTS=`which nosetests`

if [[ ! -f "$NOSETESTS" ]] ; then
    NOSETESTS=`which nosetests2`
fi

if [[ ! -f "$NOSETESTS" ]] ; then
    echo 'Cannot find nosetests or nosetests2';
else
   echo "Using $NOSETESTS"
   $NOSETESTS --with-doctest --with-coverage \
              --exe \
              --cover-package="pyautoplot" \
              --cover-tests \
              --cover-html \
              --cover-html-dir=coverage \
              --cover-erase \
              -x $@ pyautoplot . scripts
fi

