loc="/u/home/a/agauli/project-ernst/challengedata/data/590/9783590/features"
abs="/u/home/a/agauli/project-ernst/convertlibsvm"
cd $loc

for file in *
do
	echo "#!/bin/bash
#$ -o /u/home/a/agauli/joblogs
#$ -e /u/home/a/agauli/joblogs
#$ -m a
#$ -l h_data=16G,highp,h_rt=24:00:00

. /u/local/Modules/default/init/modules.sh
module load python
gunzip $loc/$file
python $abs/tsv2libsvm.py $loc/${file%???} $abs/data/${file%???????} 2 1
gzip $loc/${file%???}
"  > $file.sh
	qsub $file.sh
	rm $file.sh
done






