#!/bin/sh

logfile=testlog.txt

rm  $logfile
if touch $logfile; then
    echo "Will write results to $logfile"
else
    echo "Cannot write to log file $logfile. Proceeding without saving test results."
    logfile=/dev/null
fi


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


if test "$logfile" = "testlog.txt"; then
    echo
    echo
    echo S U M M A R Y
    echo
    cat testlog.txt |grep -e 'Ran\|OK\|FAILED\|Testing\|Error\|^    \|^  File \^\|line\|^Traceback'
fi
