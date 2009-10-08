#!/bin/bash

logfile=testlog.txt

rm  $logfile
rm .figleaf

FIGLEAF=`which figleaf`

if test "$FIGLEAF" = ""; then
    echo "Cannot find figleaf. Proceeding without test coverage analysis"
fi

if touch $logfile; then
    echo "Will write results to $logfile"
else
    echo "Cannot write to log file $logfile. Proceeding without saving test results."
    logfile=/dev/null
fi

modules=`ls *.py|grep -v "^test"`
for module in $modules; do
    echo |tee -a $logfile
    echo "*** Testing module ${module} ***"|tee -a $logfile 
    echo |tee -a $logfile
    testfile=test${module}
    if test -e $testfile; then
        
        if test "$FIGLEAF" = ""; then
            python $testfile 2>&1 |tee -a $logfile |grep -e 'Ran\|OK\|FAILED\|Testing\|Error\|^    \|^  File \^\|line\|^Traceback\| != \| !< '
        else
            figleaf $testfile 2>&1 |tee -a $logfile |grep -e 'Ran\|OK\|FAILED\|Testing\|Error\|^    \|^  File \^\|line\|^Traceback\| != \| !< '
        fi
        echo Completed testing  ${module}
    else
        echo "Error: Module $module has no associated test" |tee -a $logfile
    fi
    echo ================================================================================ >> $logfile
done

if test "$FIGLEAF" != ""; then
    figleaf2html -d coverage/ .figleaf
fi


if test "$logfile" = "testlog.txt"; then
    echo
    echo
    echo S U M M A R Y
    echo
    cat $logfile  |grep -e 'Ran\|OK\|FAILED\|Testing\|Error\|^    \|^  File \^\|line\|^Traceback\| != \| !< '
fi
