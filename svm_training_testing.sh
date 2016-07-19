#!/bin/bash
#$ -o /u/home/s/scottdet/scott/joblogs
#$ -e /u/home/s/scottdet/scott/joblogs
#$ -m a
#$ -l h_data=16G,highp,h_rt=24:00:00

head -n 450412 /u/home/s/scottdet/project-ernst/data/challengedata/chr10.tsvfinal > /u/home/s/scottdet/project-ernst/data/challengedata/chr10_testing.tsvfinal
tail -n 2252059 /u/home/s/scottdet/project-ernst/data/challengedata/chr10.tsvfinal > /u/home/s/scottdet/project-ernst/data/challengedata/chr10_training.tsvfinal

module load python
python /u/home/s/scottdet/project-ernst/data/challengedata/convert_libsvm.py /u/home/s/scottdet/project-ernst/data/challengedata/chr10_testing.tsvfinal
python /u/home/s/scottdet/project-ernst/data/challengedata/convert_libsvm.py /u/home/s/scottdet/project-ernst/data/challengedata/chr10_training.tsvfinal

/u/home/s/scottdet/project-ernst/data/libsvm-3.21/svm-train /u/home/s/scottdet/project-ernst/data/challengedata/chr10_testing_libsvm.bz2
/u/home/s/scottdet/project-ernst/data/libsvm-3.21/svm-predict /u/home/s/scottdet/project-ernst/data/challengedata/chr10_testing_libsvm.bz2 /u/home/s/scottdet/chr10_training.model /u/home/s/scottdet/project-ernst/data/challengedata/svm.output

