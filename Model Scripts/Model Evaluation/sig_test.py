import pandas as pd
from os import listdir

fs = ['../new_boot/' + item for item in listdir('../new_boot/') if '.out' in item]
main = pd.DataFrame(columns=['model','Data','accuracy_train', 'accuracy_test','pt_score', 'rt_score', 'f1t_score','severe_acc','mild_acc'])

for item in fs:
	
	if main.shape[0] == 0:
		main = pd.read_csv(item, sep='\t',names=['model','Data','accuracy_train', 'accuracy_test','pt_score', 'rt_score', 'f1t_score','severe_acc','mild_acc'])
	else:
		p2 = pd.read_csv(item, sep='\t',names=['model','Data','accuracy_train', 'accuracy_test','pt_score', 'rt_score', 'f1t_score','severe_acc','mild_acc'])
		main = pd.concat([main, p2], ignore_index=True)

tmp = []
for i,r in main.iterrows():
	if 'rules' in r.Data:
		tmp += ['W2V,TF-IDF,Rules']
	elif 'tfidf-2' in r.Data:
		tmp += ['W2V,TF-IDF']
#	elif 'vax-sideeffects' in r.Data:
#		tmp += ['Database']
	else:
		tmp += ['W2V']
main.Data = tmp


# v/l
plotter = pd.DataFrame(columns=['model','score','Class'])
j = 0
for i,r in main.iterrows():
	if r.Data == 'W2V,TF-IDF':
		plotter.loc[j] = [r.model,r.severe_acc,'v']
		j += 1
		plotter.loc[j] = [r.model,r.mild_acc,'l']	
		j += 1
	
from scipy import stats

models = plotter.model.unique()
for obj in models:
	v = []
	l = []
	get_sig = plotter[plotter.model == obj]
	for i,r in get_sig.iterrows():
		if r.Class == 'v':
			v += [r.score]
		else:
			l += [r.score]
	print(len(v),len(l),obj)
	print(obj,stats.ttest_ind(v, l)[1])
