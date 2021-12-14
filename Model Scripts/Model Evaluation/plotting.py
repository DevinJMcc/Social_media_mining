import pandas as pd
from os import listdir

fs = ['../new_boot/' + item for item in listdir('../new_boot/') if '.out' in item]
main = pd.DataFrame(columns=['model','Data','accuracy_train', 'accuracy_test','pt_score', 'rt_score', 'f1t_score','severe_acc','mild_acc'])

for item in fs:
	
#	data = pd.read_csv(item, sep='\t',names=['model','Data','accuracy_train', 'accuracy_test','pt_score', 'rt_score', 'f1t_score','severe_acc','mild_acc'])

#	if data.shape[0] != 0:
#		print(data.model.loc[0],data.Data.loc[0],data.accuracy_test.mean(),data.pt_score.mean(),data.rt_score.mean(),data.f1t_score.mean())
#		print(data.model.loc[0],data.Data.loc[0],data.accuracy_test.median(),data.pt_score.median(),data.rt_score.median(),data.f1t_score.median())
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
	elif 'vax-sideeffects' in r.Data:
		tmp += ['Database']
	else:
		tmp += ['W2V']
main.Data = tmp

main = main[main.model != 'svm']
main = main[main.model != 'random forest']
main = main[main.model != 'extra trees']

fs = main.Data.unique()
for item in fs:
	
	ms = main[main.Data == item]
	mdls = ms.model.unique()
#	print(mdls)
	for obj in mdls:
#		print(obj)
		mss = ms[ms.model == obj]
		print(mss.model.iloc[0],mss.Data.iloc[0],mss.accuracy_test.median(),mss.pt_score.median(),mss.rt_score.median(),mss.f1t_score.median())
exit()



import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.figure(figsize=(12,8))
sns.set_style("whitegrid")
ax = sns.boxplot(x="model", y="accuracy_test", hue="Data", data=main)
plt.axhline(y=0.5902, color='purple', linestyle='--', label='Baseline')
plt.ylabel('Accuracy Score',fontsize=14)
plt.xlabel('')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('model_var.pdf')
plt.close()

# v/l
plotter = pd.DataFrame(columns=['model','score','Class'])
j = 0
for i,r in main.iterrows():
	if r.Data == 'W2V,TF-IDF':
		plotter.loc[j] = [r.model,r.severe_acc,'v']
		j += 1
		plotter.loc[j] = [r.model,r.mild_acc,'l']	
		j += 1
	
sns.set_style("whitegrid")
plt.figure(figsize=(10,8))
ax = sns.boxplot(x="model", y='score', hue="Class", data=plotter)
plt.ylabel('Accuracy Score',fontsize=14)
plt.xlabel('')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('class_predi_tfidf.pdf')
