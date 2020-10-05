files=$(ls *.csv.gz)
for f in $files
do
	file=$(echo $f)
	path="."
	name=$(echo $file | cut -d. -f1)
	out="/tableize_queries.txt"
	echo $path
	echo $out
	echo $name
	../../../tabelize/tabelize.py -i "$f" -n "$name" -c >> $path$out
done