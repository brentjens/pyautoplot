#!/bin/bash
# This cript is to be called from lfe001 and will launch msplots
# instances on all non-developer compute nodes. Any arguments are passed
# directly to the msplots instances.
# cexec1 is used because it allows us to address each lce node by the same
# number as in its name. The script assumes that msplots is in the PATH
# and that the pyautoplot module is in the PYTHONPATH.

HOSTNAME=`hostname`
PATH="$PATH:/opt/cep/pyautoplot/bin"
LOG=/globaldata/inspect/launch-msplots.log

DATE=`date`
echo "" | tee -a $LOG
echo "=======================" | tee -a $LOG
echo "Date: $DATE"|tee -a $LOG
echo "$0 $@" | tee -a $LOG
echo "On machine $HOSTNAME" | tee -a $LOG

if test "$HOSTNAME" == "lhn001"; then
    cexec locus: "bash -ilc \"use LofIm; use Pythonlibs; use Pyautoplot; msplots $@\"" | tee -a $LOG

    CREATE_HTML=`which create_html`
    echo "$@" >> /globaldata/inspect/launch-msplots.log
    if test "$CREATE_HTML" == ""; then
        echo "Cannot find create_html: no HTML generated" | tee -a $LOG

    else
        echo "Creating HTML using $CREATE_HTML" | tee -a $LOG
        command="$CREATE_HTML $@"
        echo "$command"| tee -a $LOG
        result=`$command`
        exit_status="$?"
        if test "$exit_status" == "0"; then
            echo "HTML Created successfully" | tee -a $LOG
        else 
            echo "Problem creating HTML overview for $@." | tee -a $LOG
            echo "Exit status: $exit_status" | tee -a $LOG
            echo "$result" | tee -a $LOG
        fi
    fi
else
    cexec1 lce:1-54,64-72 "bash -ilc \"use LofIm;use Pythonlibs; use Pyautoplot; msplots $@\"" | tee -a $LOG
fi

DATE_DONE=`date`
echo "Done at $DATE_DONE" | tee -a $LOG
