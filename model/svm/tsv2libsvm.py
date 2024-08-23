#!/usr/bin/env python

"""
Convert TSV file to libsvm format.
Put -1 as label index (argv[3]) if there are no labels in your file.
Expecting no headers. If present, headers can be skipped with argv[4] == 1.

"""

import sys
import csv
from collections import defaultdict

def construct_line( label, line ):
        new_line = []
        if float( label ) == 0.0:
                label = "0"
        new_line.append( label )
        for i, item in enumerate( line ):
		item = int(item)
                if item == '' or float( item ) == 0.0:
                        continue
                new_item = "%s:%s" % ( i + 1, item )
                new_line.append( new_item )
        new_line = " ".join( new_line )
        new_line += "\n"
        return new_line

# ---

input_file = sys.argv[1]
output_file = sys.argv[2]

try:
        label_index = int( sys.argv[3] ) 
except IndexError:
        label_index = 0

try:
        skip_headers = sys.argv[4]
except IndexError:
        skip_headers = 0

i = open( input_file, 'rb' )
o = open( output_file, 'wb' )

reader = csv.reader(i, delimiter = '\t')

if skip_headers:
        headers = reader.next()

bound = 0
unbound = 0
ambig = 0
null = 0
nullunbound = 0
nullbound = 0
nullamb = 0


for line in reader:
        runindex = label_index+4096
        del line[0:4]
        label = line.pop(runindex)
        if label == "U":
                unbound+=1
                labelnum = "0"
        elif label == "B":
                bound += 1
                labelnum = "1"
        elif label == "A":
                ambig += 1
                continue
        del line[4096:]
        new_line = construct_line( labelnum, line )
        if len(new_line) == 2:
                null+=1
                if labelnum == "0":
                        nullunbound +=1
                elif labelnum == "1":
                        nullbound += 1
                elif label == "A":
                        nullamb +=1
                continue
        o.write( new_line )
        runindex = label_index


stat = open( output_file + "stat", 'wb' )
stat.write("chr\tbound\tunbound\tambig\tnull\tnullunbound\tnullbound\tnullamb\n")
stat.write(output_file +"\t"+ str(bound)+"\t"+ str(unbound) + "\t" + str(ambig) + "\t" + str(null) + "\t" + str(nullunbound) + "\t" + str(nullbound) + "\t" + str(nullamb) + "\n")
