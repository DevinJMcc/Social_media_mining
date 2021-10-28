dir=(/home/FCAM/dmcconnell/Social_media_mining/data_files/word*.csv)

for ((i=0; i<${#dir[@]}; i++)); do
	for j in {0..6}; do
		echo 'python3 model_search.py '${dir[$i]} ${j} >> submit_classification_jobs.sh
	done
done
