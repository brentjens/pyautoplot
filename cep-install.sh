#!/bin/sh
rsync -avz --delete ./ brentjens@dop95.astron.nl:pyautoplot/ && lfc lhn001 'rsync -avz --delete brentjens@dop95.astron.nl:pyautoplot/ ./pyautoplot/'
