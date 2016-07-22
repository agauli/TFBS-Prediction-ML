#!/bin/bash
#$ -o /u/home/s/scottdet/scott/joblogs
#$ -e /u/home/s/scottdet/scott/joblogs
#$ -m a
#$ -l h_data=16G,highp,h_rt=24:00:00

awk '{if ($1 == '1') print $1, $2}' ./project-ernst/data/challengedata/svm_testing_data/chr10_libsvm.bz2 > chr10bound_libsvm.bz2
awk '{if ($1 == '1') print $1, $2}' ./project-ernst/data/challengedata/svm_testing_data/chr11_libsvm.bz2 > chr11bound_libsvm.bz2
cat chr10bound_libsvm.bz2 chr11bound_libsvm.bz2 > boundTraining_libsvm.bz2
rm chr10bound_libsvm.bz2
rm chr11bound_libsvm.bz2

wc -l boundTraining_libsvm.bz2		# I did this part manually, just checked how many were bound so I could take 1/6 of that for testing

./project-ernst/data/libsvm-3.21/tools/subset.py -s 1 chr10_libsvm.bz2 [1/2 size of boundTraining_libsvm] chr10unbound_libsvm.bz2
./project-ernst/data/libsvm-3.21/tools/subset.py -s 1 chr11_libsvm.bz2 [1/2 size of boundTraining_libsvm] chr11unbound_libsvm.bz2
cat chr10unbound_libsvm.bz2 chr11unbound_libsvm.bz2 > unboundTraining_libsvm.bz2
rm chr10unbound_libsvm.bz2
rm chr11unbound_libsvm.bz2

wc -l unboundTraining_libsvm.bz2		# Make sure bound and unbound are same size

cat boundTraining_libsvm.bz2 unboundTraining_libsvm.bz2 > equalTraining_libsvm.bz2
rm boundTraining_libsvm.bz2
rm unboundTraining_libsvm.bz2

wc -l equalTraining_libsvm.bz2		# 1/6 size used for testing as with libsvm examples

awk '{if ($1 == '1') print $1, $2}' ./project-ernst/data/challengedata/svm_testing_data/chr12_libsvm.bz2 > chr12bound_libsvm.bz
./project-ernst/data/libsvm-3.21/tools/subset.py -s 1 chr12bound_libsvm.bz2 [1/12 size of equalTraining_libsvm.bz2] chr12bound_subset_libsvm.bz2		# 1/12 because we want testing to be 1/6 size of training
rm chr12bound_libsvm.bz2

./project-ernst/data/libsvm-3.21/tools/subset.py -s 1 chr12_libsvm.bz2 [1/12 size of boundTraining_libsvm] chr12unbound_libsvm.bz2	

cat chr12bound_subset_libsvm.bz2 chr12unbound_libsvm.bz2 > equalTesting_libsvm.bz2
rm chr12bound_subset_libsvm.bz2
rm chr12unbound_libsvm.bz2

./project-ernst/data/libsvm-3.21/svm-train ./equalTraining_libsvm.bz2
./project-ernst/data/libsvm-3.21/svm-predict ./equalTesting_libsvm.bz2 ./equalTraining_model_libsvm.bz2 svm_output.tsv	



