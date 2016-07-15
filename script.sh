#!/bin/sh
mkdir $1chr
mv $1 $1chr
cd $1chr
awk '{print >> $1}' $1
cut -f 1-3 chr > header1.txt
cut -f 4- chr > header2.txt
#celltype="$(awk'{$1=$2=$3="";print $0}' chr)"

mv $1 ..
mv chr ..

mv header1.txt ..
mv header2.txt ..

 
for file in *
do 
	if test -f "$file"
	then
		mv $file $file.tsv
		head ../header1.txt > $file.tsva
		head ../header2.txt > $file.tsvb
		cut -f 1-3 $file.tsv >> $file.tsva
		cut -f 4- $file.tsv >> $file.tsvb
		bedtools getfasta -fo output.tsv -tab -fi ../hg19.genome.fa -bed $file.tsv -split -name
		mv output.tsv $file.tsv
		python ../extractfeatures.py $file.tsv
		paste "$file.tsva" "$file.tsvf" "$file.tsvb" > $file.tsvfinal
		echo "FInished pasting"
		rm $file.tsva 
		rm $file.tsvb
		rm $file.tsvf
		rm $file.tsv
	fi
done
cd ..
rm header1.txt
rm header2.txt
rm chr
echo "File conversion finished"
