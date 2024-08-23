loc_svm_divided="/u/home/a/agauli/project-ernst/convertlibsvm/ctcflibsvm"
loctest_train="/u/home/a/agauli/project-ernst/convertlibsvm/datamerget"
chrtest=(chr9 chr7 chrX chr6 chr10 )
chrtrain=(chr2 chr3 chr4 chr5 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr22)
ratios=(1 2 4 8 16 32 64 128)

cd $loctest_train
touch test.tsv
touch train.tsv

for testfile in "${chrtest[@]}"
do
	cat $loc_svm_divided/$testfile* >> $loctest_train/test.tsv
done

for trainfile in "${chrtrain[@]}"
do
	cat $loc_svm_divided/$trainfile* >> $loctest_train/train.tsv
done

awk '{if ($1 == "1") print substr($0, index($1, $3))}' $loctest_train/train.tsv > $loctest_train/bound.tsv
awk '{if ($1 == "-1") print substr($0, index($1, $3))}' $loctest_train/train.tsv > $loctest_train/unbound.tsv

bound=$(wc -l < $loctest_train/bound.tsv)
unbound=$(wc -l < $loctest_train/unbound.tsv)

for (( i = 0 ; i < "${#ratios[@]}"; i++ )) 
do
    DMGLIST2[$i]=$((${ratios[$i]}*$bound))
done

for e in "${DMGLIST2[@]}"
do 
	echo "#!/bin/bash
#$ -o /u/home/a/agauli/joblogs
#$ -e /u/home/a/agauli/joblogs
#$ -m a
#$ -l h_data=16G,highp,h_rt=24:00:00
. /u/local/Modules/default/init/modules.sh
module load python
head -n  $e  $loctest_train/unbound.tsv > $loctest_train/unbound$e
echo $unbls
cat $loctest_train/bound.tsv $loctest_train/unbound$e > $loctest_train/merged$e
shuf $loctest_train/merged$e > $loctest_train/mergedsuffled$e
python /u/home/a/agauli/project-ernst/convertlibsvm/libsvm-3.21/python/plotroc.py $loctest_train/mergedsuffled$e" > $loctest_train/final$e.sh
qsub $loctest_train/final$e.sh
done
