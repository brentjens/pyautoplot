#!/bin/bash
# This cript is to be called from lfe001 and will launch msplots
# instances on all non-developer compute nodes. Any arguments are passed
# directly to the msplots instances.
# cexec1 is used because it allows us to address each lce node by the same
# number as in its name. The script assumes that msplots is in the PATH
# and that the pyautoplot module is in the PYTHONPATH.

HOSTNAME=`hostname`
PATH="$PATH:/opt/cep/pyautoplot/bin"
INSPECT_ROOT=/globaldata/inspect
LOG=$INSPECT_ROOT/launch-msplots.log


#Time to wait for stuck processes before killing them
export ALARMTIME=300
 
PARENTPID=$$
GLOBAL_ARGS=$@
COMMAND_NAME='msplots $@'


create_html_fn() {
    CREATE_HTML=`which create_html`
    echo "$GLOBAL_ARGS" | tee -a $LOG
    if test "$CREATE_HTML" == ""; then
        echo "Cannot find create_html: no HTML generated" | tee -a $LOG

    else
        echo "Creating HTML using $CREATE_HTML" | tee -a $LOG
        command="$CREATE_HTML $GLOBAL_ARGS"
        echo "$command"| tee -a $LOG
        result=`$command`
        exit_status="$?"
        if test "$exit_status" == "0"; then
            echo "HTML Created successfully" | tee -a $LOG
        else 
            echo "Problem creating HTML overview for $GLOBAL_ARGS." | tee -a $LOG
            echo "Exit status: $exit_status" | tee -a $LOG
            echo "$result" | tee -a $LOG
        fi
    fi
}

exit_timeout() {
    echo "TIMEOUT : killing cexec ($CEXEC_PID)" | tee -a $LOG
    child_pids=`ps -o user,pid,ppid,command ax |grep "$COMMAND_NAME"|grep -v grep|awk '{print $2}'`
    for pid in $child_pids; do
        pkill -TERM -P $pid > /dev/null 2>&1
        done
    pkill -TERM -P $CEXEC_PID >/dev/null 2>&1
    create_html_fn
    DATE_DONE=`date`
    echo "Done at $DATE_DONE" | tee -a $LOG
    exit
}

DATE=`date`
echo "" | tee -a $LOG
echo "=======================" | tee -a $LOG
echo "Date: $DATE"|tee -a $LOG
echo "$0 $@" | tee -a $LOG
echo "On machine $HOSTNAME" | tee -a $LOG

if test "$HOSTNAME" == "lhn001"; then
    for sas_id in $@; do
        mkdir $INSPECT_ROOT/$sas_id $INSPECT_ROOT/HTML/$sas_id
        done

    #Prepare to catch SIGALRM, call exit_timeout
    trap exit_timeout SIGALRM
    
    cexec locus: "bash -ilc \"use LofIm; use Pythonlibs; use Pyautoplot; $COMMAND_NAME\"" &
    CEXEC_PID=$!
    #Sleep in a subprocess, then signal parent with ALRM
    (sleep $ALARMTIME; kill -ALRM $PARENTPID) &
    #Record PID of subprocess
    ALARMPID=$!
    
    #Wait for child processes to complete normally
    wait $CEXEC_PID
 
    #Tidy up the Alarm subprocess
    kill $ALARMPID
    
    create_html_fn

else
    cexec1 lce:1-54,64-72 "bash -ilc \"use LofIm;use Pythonlibs; use Pyautoplot; $COMMAND_NAME\"" | tee -a $LOG
fi

DATE_DONE=`date`
echo "Done at $DATE_DONE" | tee -a $LOG
