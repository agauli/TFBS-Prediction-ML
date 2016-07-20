import sys
chr_sequence_binding= open(str(sys.argv[1]), 'r')
chr_features_libsvm = open("%s" % str(sys.argv[1])[:-9]+ "_libsvm.bz2", 'w')
chr_sequence_binding.readline()
for line in chr_sequence_binding:
        line = line.rstrip("\n")
        splitline = line.split("\t")
        string = ""
        binding = splitline[4101]
        if binding == "U":
                string += "0"
        elif binding == "B":
                string += "1"
        else:
                string += "2"

        string += " "
        for num in range(4, 4100):
                string += splitline[num]
        string +="\n"
        chr_features_libsvm.write(string)
chr_features_libsvm.close()

