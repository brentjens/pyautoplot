#!/bin/bash

PYTHONPATH="`pwd`:$PYTHONPATH"
TESTOUTPUT=test-output

rm -rf $TESTOUTPUT
mkdir $TESTOUTPUT

python make_cluster_dirs.py

NOSETESTS=`which nosetests`

if [[ ! -f "$NOSETESTS" ]] ; then
    NOSETESTS=`which nosetests2`
fi
if [[ ! -f "$PYLINT" ]] ; then
    PYLINT=`which pylint2`
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

# echo ''
# echo '  *** Pylint output ***'
# echo ''
# 
# if [[ ! -f "$PYLINT" ]] ; then
#     echo 'Cannot find pylint';
# else
#     $PYLINT --output-format=colorized --reports=n  --disable=C0103 pyautoplot scripts/create_html;
# fi
# 
# echo ''
