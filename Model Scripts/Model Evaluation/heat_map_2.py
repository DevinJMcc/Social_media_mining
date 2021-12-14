from os import listdir
import numpy as np
t1 = 'new_test2'
t2 = 'new_test2.1'
t3 = 'test2'
t4 = 'test2.1'

t1_f = [t1+'/'+item for item in listdir(t1) if '.out' in item] 
t2_f = [t2+'/'+item for item in listdir(t2) if '.out' in item]
#t3_f = [t3+'/'+item for item in listdir(t3) if '.out' in item]
#t4_f = [t4+'/'+item for item in listdir(t4) if '.out' in item]

dict = {}
for item in t1_f:
	ff = open(item,'r')
	line = ff.readline().strip().split('\t')
	if 'decision tree' in line:
		n2 = line[1].split('/')[-1]
		if n2 not in dict:
			dict[n2] = [float(line[-1]),item]
		elif n2 in dict and float(line[-1]) > dict[n2][0]:
			dict[n2] = [float(line[-1]),item]

for item in t2_f:
#	print(item)
	ff = open(item,'r')
	line = ff.readline().strip().split('\t')
	if 'decision tree' in line:
		n2 = line[1].split('/')[-1]
		if n2 not in dict:
			dict[n2] = [float(line[-1]),item]
		elif n2 in dict and float(line[-1]) > dict[n2][0]:
			dict[n2] = [float(line[-1]),item]
#print(dict)
#exit()
'''
for item in t3_f:
	ff = open(item,'r')
	line = ff.readline().strip().split('\t')
	if 'decision tree' in line:
		n2 = line[1].split('/')[-1]
		if n2 not in dict:
			dict[n2] = [float(line[-1]),item]
		elif n2 in dict and float(line[-1]) > dict[n2][0]:
			dict[n2] = [float(line[-1]),item]

for item in t4_f:
	ff = open(item,'r')
	line = ff.readline().strip().split('\t')
	if 'decision tree' in line:
		n2 = line[1].split('/')[-1]
		if n2 not in dict:
			dict[n2] = [float(line[-1]),item]
		elif n2 in dict and float(line[-1]) > dict[n2][0]:
			dict[n2] = [float(line[-1]),item]
'''

# get importance
feats = {}
for item in dict:
	error_file = dict[item][1].split('.out')[0] + '.err'
	ef = open(error_file)
	x = []
	save = 0
	count = 0
	for line in ef:
		if 'Feature ranking' in line:
			save = 1
		if save == 1 and 'Feature ranking' not in line :
			x += [line]
			count += 1
		if count == 316:
			break
	feats[item] = x
print(feats.keys())
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x_keys = ['word_embedings-Wiki-tfidf-2.csv', 'word_embedings-APNews-tfidf-2.csv', 'word_embedings-Wiki-2.csv', 'word_embedings-APNews-2.csv', 'word_embedings-APNews-tfidf-rules-2.csv', 'word_embedings-Wiki-tfidf-rules-2.csv']
xs = ['verified','sentiment_nan','sentiment_Negative','favourites_count','friends_count','followers_count','statuses_count','sentiment_Neutral','listed_count','favorite_count','sentiment_Positive','retweet_count','Fear','Sad','Suprise','Angry','Happy']

xxx = []
for item in x_keys:
	xx = [0 for i in range(len(xs)+1)]
	for obj in feats[item]:
		line = obj.strip().split(' ')
		if line[1] == xs[0]:
			xx[0] = float(line[2])
		elif line[1] == xs[1]:
			xx[1] = float(line[2])
		elif line[1] == xs[2]:
			xx[2] = float(line[2])
		elif line[1] == xs[3]:
			xx[3] = float(line[2])
		elif line[1] == xs[4]:
			xx[4] = float(line[2])
		elif line[1] == xs[5]:
			xx[5] = float(line[2])
		elif line[1] == xs[6]:
			xx[6] = float(line[2])
		elif line[1] == xs[7]:
			xx[7] = float(line[2])
		elif line[1] == xs[8]:
			xx[8] = float(line[2])
		elif line[1] == xs[9]:
			xx[9] = float(line[2])
		elif line[1] == xs[10]:
			xx[10] = float(line[2])
		elif line[1] == xs[11]:
			xx[11] = float(line[2])
		elif line[1] == xs[12]:
			xx[12] = float(line[2])
		elif line[1] == xs[13]:
			xx[13] = float(line[2])
		elif line[1] == xs[14]:
			xx[14] = float(line[2])
		elif line[1] == xs[15]:
			xx[15] = float(line[2])
		elif line[1] == xs[16]:
			xx[16] = float(line[2])
		else:
			print(line)	
			xx[17] += float(line[2])
	xxx += [xx]
plotting = np.array(xxx)

# Remove the columns with none zeros 
remove = []
for i in range(plotting.shape[1]):
    x = 0
    for item in plotting[:,i]:
        if item == 0.0:
            x = 1
        else:
            x = 0
            break
    if x == 1:
        remove += [i]


plotting = np.delete(plotting,remove,1)
print (plotting)
keep = []
for i in range(len(xs)):
    if i not in remove:
        keep += [xs[i]]

keep += ['W2V']
print(keep)
y = ['Wiki,TF-IDF', 'APNews,TF-IDF','Wiki','APNews','APNews,TF-IDF,Rules','Wiki,TF-IDF,Rules']


import seaborn as sns


fig, ax = plt.subplots(figsize=(8,8))
ax = sns.heatmap(plotting,xticklabels=keep,yticklabels=y,square=True,linewidths=0.3,cmap='gist_stern')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor",fontsize=12)
plt.yticks(fontsize=13)
sns.set(font_scale=3)

# plt.show()
plt.tight_layout()

plt.savefig('heatmap_feat_import-2.pdf')

