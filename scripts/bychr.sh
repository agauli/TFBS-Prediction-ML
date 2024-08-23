a=$1
abs=$(pwd)
mkdir -p $abs/features
mv $a $abs/features
cd $abs/features
awk '{print >> $1}' $abs/features/$a
absheader=$abs
cut -f 1-3 chr > $absheader/header1.txt
cut -f 4- chr > $absheader/header2.txt
mv $abs/features/$a $abs/
rm $abs/features/chr
abs=$abs/features


for tospl in *
do
	echo "#!/bin/bash
#$ -o /u/home/a/agauli/joblogs
#$ -e /u/home/a/agauli/joblogs
#$ -m a
#$ -l h_data=16G,highp,h_rt=24:00:00
	gzip $tospl
" > $gzip$tospl.sh
	qsub $gzip$tospl.sh
	rm $gzip$tospl.sh
done




for file in *
do
  	if test -f "$file"
        then
            	echo "#!/bin/bash
#$ -o /u/home/a/agauli/joblogs
#$ -e /u/home/a/agauli/joblogs
#$ -m a
#$ -l h_data=16G,highp,h_rt=24:00:00
. /u/local/Modules/default/init/modules.sh
module load bedtools
mv $abs/$file $abs/$file.tsv
head $absheader/header1.txt > $abs/$file.tsva
head $absheader/header2.txt > $abs/$file.tsvb
cut -f 1-3 $abs/$file.tsv >> $abs/$file.tsva
cut -f 4- $abs/$file.tsv >> $abs/$file.tsvb
bedtools getfasta -fo $absheader/$file.tsv -tab -fi /u/home/a/agauli/project-ernst/challengedata/hg19.genome.fa -bed $abs/$file.tsv -split -name
mv $absheader/$file.tsv $abs/$file.tsv
module load python
python /u/home/a/agauli/project-ernst/challengedata/extractfeatures.py $abs/$file.tsv
paste $abs/$file.tsva $abs/$file.tsvf $abs/$file.tsvb > $abs/$file.tsvfinal
				

                echo "Finished pasting"
                rm $abs/$file.tsva
                rm $abs/$file.tsvb
                rm $abs/$file.tsvf
                rm $abs/$file.tsv" > $1_$file.sh
                qsub $1_$file.sh
                rm $1_$file.sh
        fi
done
