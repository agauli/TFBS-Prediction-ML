# extractfeatures.py
# created by: Ashish Gauli, 12 July 2016

# Execution: python extractfeatures.py filename

# Extracts count-kmers features from bedtools getfasta tsv output file.
# Note: The tsv file from bedtools was obtained as followed:
# bedtools getfasta -fo output.tsv -tab -fi hg19.genome.fa -bed Small.train.labels.tsv -split -name


# INPUT FILE FORMAT : filename.tsv
# INPUT LINE FORMAT : "binding\tsequence" ~ "U   ATGCGGSGDD"
# OUTPUT LINE FORMAT : "sequence, kmers"~ "AATA..G\t1\t2\t3..\t(4^k)"
# OUTPUT FILE FORMAT : filename_features.tsv

import sys
# This functions create all possible kmers ATGC of length k.
# Input: 1)k : a integer that represents the length of kmer.
# Output: a list of sorted kmers.

def createkmers(k):
    pools = map(tuple, ('ATGC',)) * k
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    i = 0
    while i < len(result):
        result[i] = ''.join(result[i])
        i += 1
    result.sort()
    return result

# This function counts the occurance of kmers in a genetic sequence.
# Input: 1) sequence: a string of sequence.
#        2) k : a integer that represents the length of kmer.
# Output: a list, containing counts as integers, alphabetically ordered wrt to kmers.


def countkmer(sequence, k):
    kmers = {}
    for prod in createkmers(k):
        kmers[''.join(prod)] = 0
    n = len(sequence)
    for i in range(0, n-k+1):
        kmer = (sequence[i:i + k]).upper()
        if "N" not in kmer:
            kmers[kmer] += 1
    return [v for x, v in sorted(kmers.iteritems())]


def convert(k):
    chr_sequence_binding = open(str(sys.argv[1]), 'r')
    chr_features_binding = open("%s" % str(sys.argv[1]).strip(".tsv")\
                                + "_features.tsv", 'w')
    header = "sequence\t"
    kmers = createkmers(k)
    for prod in kmers:
        header += ''.join(prod) + "\t"
    header += "\n"
    chr_features_binding.write(header)

    for line in chr_sequence_binding:
        line = line.rstrip("\n")
        splitline = line.split("\t")
        sequence = splitline[1]
        chr_feature = sequence + "\t"
        for counts in countkmer(sequence, k):
            chr_feature += (str(counts) + "\t")
        chr_feature += "\n"
        chr_features_binding.write(chr_feature)
    chr_features_binding.close()


convert(6)
