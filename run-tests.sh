#!/bin/sh

modules="angle pyautoplot"
for module in $modules; do
    echo
    echo "*** Testing module ${module} ***"
    echo
    python test${module}.py
    echo Completed testing  ${module}
    echo
    echo ================================================================================ 
done
