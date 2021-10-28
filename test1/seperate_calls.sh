SCRIPT="submit_classification_jobs.sh"
SCRIPT_PARAMETERS=""
CLUSTERIZE_LOC=""
CLUSTERIZE_PARAM=""

rm -f  x*SLL

split -l 1 --additional-suffix SLL $SCRIPT

i=0
for f in "x"*"SLL"; do
	array[ $i ]="$f"
	(( i++ ))
done
