for i  in 0 1 2 3 4
do
	for model in svm rf lr bt
	do
		python3 model.py -t list/train_$i.csv -m $model -d breath &
	done 
done
