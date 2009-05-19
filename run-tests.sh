#!/bin/sh

logfile=testlog.txt

rm  $logfile

modules="angle pyautoplot"
for module in $modules; do
    echo >> $logfile
    echo "*** Testing module ${module} ***">>$logfile 
    echo >> $logfile
    python test${module}.py 2>>$logfile
    echo Completed testing  ${module}
    echo ================================================================================ >> $logfile
done

echo
echo
echo S U M M A R Y
echo
cat testlog.txt |grep -e 'Ran\|OK\|FAILED\|Testing'
