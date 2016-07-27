#loc_svm_divided="/u/home/a/agauli/project-ernst/convertlibsvm/ctcflibsvm"
#loctest_train="/u/home/a/agauli/project-ernst/convertlibsvm/datamerget"
#chrtest=(9 7 X 6 )
#chrtrain=(2 3 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 22 X)
#ratios=(1 1.5 2 2.5 3 3.5 4)

loc_svm_divided="/u/home/a/agauli/project-ernst/convertlibsvm/test"
loctest_train="/u/home/a/agauli/project-ernst/convertlibsvm/datamerget"
chrtest=(chr6)
chrtrain=(chr7 chr9 chrX)
ratios=(1 2)

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

#$totallines=wc -l $loctest_train$train.tsv

awk '{if ($1 == "1") print substr($0, index($1, $3))}' $loctest_train/train.tsv > $loctest_train/bound.tsv
awk '{if ($1 == "0") print substr($0, index($1, $3))}' $loctest_train/train.tsv > $loctest_train/unbound.tsv

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
head -n $e $loctest_train/unbound.tsv  > $loctest_train/chr_unbound$e
cat $loctest_train/bound.tsv $loctest_train/chr_unbound$e > $loctest_train/chrmerged$e
shuf $loctest_train/chrmerged$e > $loctest_train/chrmersuf$e
python /u/home/a/agauli/project-ernst/convertlibsvm/libsvm-3.21/python/plotroc.py $loctest_train/chrmersuf$e" > $loctest_train/$e.sh
qsub $loctest_train/$e.sh
done



