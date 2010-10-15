#!/bin/sh
rsync -avz --delete ./ brentjens@dop95.astron.nl:pyautoplot/ && lfc lce030 'rsync -avz --delete brentjens@dop95.astron.nl:pyautoplot/ ./pyautoplot/'
