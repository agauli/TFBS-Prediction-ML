# TFBS-Prediction-ML

Predicting in-vivo transcription factor binding sites (TFBS) from DNA sequence. 

This was an internship project from around 2016. It is kept as an archive and is not actively
maintained.

## What it does

Given a window of human genomic DNA, the models predict whether a transcription factor
(here **CTCF**) is bound to it.

1. **Feature extraction**: each sequence is converted into 6-mer counts (4^6 = 4096 features).
2. **Data prep**: features are split by chromosome, converted to LibSVM format, and unbound
   examples are subsampled to balance against the rarer bound examples.
3. **Models**: a linear SVM (LibSVM) and logistic regression (Theano and TensorFlow).
4. **Evaluation**: train and test on different chromosomes and compare with ROC curves and AUC.

## Layout

```
scripts/        feature extraction (extractfeatures.py) and per-chromosome splitting (*.sh)
model/svm/      TSV to LibSVM conversion, SVM training, and ROC plotting
Logistic/       logistic regression: Theano, Theano (chunked/multiprocess), TensorFlow
data/           CTCF_subsample.tsv: 500-row sample (chr, start, stop, sequence, 4096 k-mer counts)
equal_samples.sh, svm_train_test.sh   class-balancing and SVM train/test helpers
```

## Pipeline

```
bedtools getfasta -> extractfeatures.py -> chrdivided.sh / bychr.sh
                                                  |
                              tsv2libsvm.py <-----+
                                    |
            svm_train_test.sh / plotroc.sh      Logistic/*.py
```

Feature extraction input comes from `bedtools getfasta -tab -fi hg19.genome.fa -bed <labels.tsv>`.

## Running on the cluster

The genome-wide feature tables are too large for one machine, so the heavy steps were run as
batch jobs on UCLA's [Hoffman2 cluster](https://www.hoffman2.idre.ucla.edu/About/System-overview.html),
a Linux compute cluster managed by UCLA OARC (formerly IDRE). Jobs are dispatched by a Grid
Engine scheduler (Univa Grid Engine in this era). Each script writes a small job file and submits
it with `qsub`, so every chromosome or chunk runs as its own parallel job:

| Where | What it parallelizes |
|---|---|
| `scripts/chrdivided.sh` | splits features by chromosome, then one job per 200k-line chunk (loads `bedtools`) |
| `scripts/bychr.sh` | one job per chromosome file (gzip, then feature processing) |
| `model/svm/tsv2libsvm.sh` | one job per chromosome file: `gunzip`, `tsv2libsvm.py`, `gzip` |
| `model/svm/plotroc.sh` | one SVM train/test job per bound:unbound ratio (1x to 128x) |

The header on each job file means:

```
#$ -o / -e           where stdout / stderr logs go (the joblogs/ directory)
#$ -m a              email when a job aborts
#$ -l h_data=16G     memory per core (h_data is per core, not per job)
#$ -l highp          run on group-owned nodes (higher priority, runs up to 14 days)
#$ -l h_rt=24:00:00  hard wall-clock limit of 24 hours
. /u/local/Modules/default/init/modules.sh
module load python  # or bedtools: software comes from the environment-modules system
```

To run this elsewhere, replace the `qsub` and `#$` lines with your scheduler's equivalent (for
example Slurm `sbatch` and `#SBATCH`), and update the `module load` lines and `/u/home/...` paths.
Hoffman2 has since moved to Altair Grid Engine, with a Slurm preview cluster, so the current
[Hoffman2 docs](https://www.hoffman2.idre.ucla.edu/Using-H2/Computing/Computing.html) apply if you
rerun it there.

## Requirements and caveats

- Python 2, Theano, TensorFlow 1.x, pandas, scikit-learn, matplotlib
- LibSVM 3.21 and bedtools
- The pipeline was run on UCLA's Hoffman2 cluster (see below). The shell scripts contain
  hardcoded paths (`/u/home/...`). Edit them before use elsewhere.
- The full challenge data is not included, only the small sample in `data/`.
- Known issues: some scripts are unfinished or have bugs (e.g. `train_test_plotroc.sh`,
  `equal_samples.sh`, `convert_libsvm.py`), and the logistic scripts split randomly rather than
  by chromosome, which inflates their AUC.

## Author

Ashish Gauli
