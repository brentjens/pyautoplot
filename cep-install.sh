#!/bin/sh

USER=$1

function lfc {
    ssh -AYCtt  ${USER}@portal.lofar.eu  "ssh -AYCtt  $1 'bash -ilc \"$2\"'"
}

rsync -a --no-group -vz --delete -e "ssh -A ${USER}@portal.lofar.eu ssh" ./ ${USER}@lhn001:pyautoplot/

