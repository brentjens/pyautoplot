#!/bin/sh

logfile=testlog.txt

rm  $logfile

modules=`ls *.py|grep -v "^test"`
for module in $modules; do
    echo >> $logfile
    echo "*** Testing module ${module} ***">>$logfile 
    echo >> $logfile
    testfile=test${module}
    if test -e $testfile; then
        python $testfile 2>>$logfile
        echo Completed testing  ${module}
    else
        echo "Error: Module $module has no associated test" |tee -a $logfile
    fi
    echo ================================================================================ >> $logfile
done

echo
echo
echo S U M M A R Y
echo
cat testlog.txt |grep -e 'Ran\|OK\|FAILED\|Testing\|Error\|^    \|^  File \^\|line\|^Traceback'
