chr_feature_binding= open(str(sys.argv[1]), 'r')
chr_features_libsvm = open("%s" % str(sys.argv[1]).strip(".tsvf")\
                                + "_libsvm.bz2", 'w')
chr_feature_binding.readline()
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

	chr_features_libsvm.write(string)

chr_features_libsvm.close()
