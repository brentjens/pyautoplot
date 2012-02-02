#!/bin/sh

USER=$1

rsync -avz --delete -e "ssh -A ${USER}@portal.lofar.eu ssh" ./ ${USER}@lhn001:pyautoplot/
