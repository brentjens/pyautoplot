#!/bin/sh
rsync -avz --delete ./ brentjens@dop95:pyautoplot/ && lfc lce001 'rsync -avz --delete brentjens@dop95.astron.nl:pyautoplot/ ./pyautoplot/'
