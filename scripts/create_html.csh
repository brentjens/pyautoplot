#!/bin/csh
#
# Create HTMl index for autoplots. 
#
# Usage: create_html.csh <ObsID>
#

cd /globaldata/inspect/HTML
mkdir $1
mkdir $1/SBpages

echo "<html> <head> <title> L$1 </title></head><body> " > $1/index.html
echo "<h1> L$1 </h1> " >> $1/index.html
echo " <a href=../index.html>Back to Projects</a> " >> $1/index.html
echo " </br> " >> $1/index.html

echo Processing directory ../$1/

set bands=`ls -1 ../$1/*flagged-mean.png`
foreach flaggedmean ($bands)
	echo $flaggedmean
	set SB=`echo $flaggedmean | cut -d_ -f3`
	echo " <a href=SBpages/"$SB".html>"$SB"</a> " >> $1/index.html
	echo "<html> <head> <title> $SB </title></head><body> " > $1/SBpages/$SB.html
	echo "<h1> L$1 $SB </h1> " >> $1/SBpages/$SB.html
	echo " <a href=../../index.html>Back to Projects</a> " >> $1/SBpages/$SB.html
	echo " </br> " >> $1/SBpages/$SB.html
	foreach band ($bands)
	    set bnd=`echo $band | cut -d_ -f3`
	    echo " <a href="$bnd".html>"$bnd"</a> " >> $1/SBpages/$SB.html
	end
	foreach img (../$1/*$SB*.png)
	    echo "<img src="../../$img" width="1480"/>" >> $1/SBpages/$SB.html
	end
end

echo "<html> <head> <title> Projects </title></head><body> " > index.html
echo "<h1> Projects </h1> " >> index.html
set projects=`ls -1 | egrep '1|2|3|4|5|6|7' | sort -r`
foreach project ($projects)
	echo " <a href="$project"/index.html>"$project"</a> " >> index.html
	echo " </br> " >> index.html
end
