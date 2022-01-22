import matplotlib.pyplot as plt

import pickle
import numpy as np

c_short = "acc"

cases_short = ["nom", "gen", "dat", "acc", "voc", "loc", "ins"]

import os

#nom_acc = np.asarray(pickle.load(open(f"tf_data/{c1_short}-{c2_short}.pkl", "rb")))

xs = list(range(13))
all_datas = {}
for c_short in cases_short:
	datas = {}
	for cur_short in cases_short:
		if cur_short == c_short: continue
		pathname = f"new_data/new-{c_short}-{cur_short}.pkl"
		if not os.path.exists(pathname):
			pathname = f"new_data/new-{cur_short}-{c_short}.pkl"
		datas[cur_short] = np.asarray(pickle.load(open(pathname, "rb")))

	nom_all = np.asarray(pickle.load(open(f"new_data/new-{c_short}-all.pkl", "rb")))
	nom_rnd = np.asarray(pickle.load(open(f"new_data/new-{c_short}-random.pkl", "rb")))
	for cur_short in datas:
		plt.plot(xs, nom_rnd-datas[cur_short], "o-", label=f"{c_short}-{cur_short}")
	plt.plot(xs, nom_rnd-nom_all, "o-", label="same roots")
	#plt.plot(xs, nom_rnd-nom_rnd, "o-", label="completely random")
	plt.legend()
	plt.show()

	datas[f"{c_short}-rnd"] = nom_rnd
	datas[f"{c_short}-all"] = nom_all
	all_datas[c_short] = datas

import numpy as np
rows = [['----'] + list(map(lambda x: x + "\t", list(all_datas)))
	+ ['all\t']]

layers = []

for c_short in all_datas:
	row = [c_short]
	datas = all_datas[c_short]
	for cur_short in all_datas:
		if cur_short == c_short:
			row.append("----")
			continue
		all_diff = datas[f"{c_short}-rnd"]-datas[cur_short]
		max_diff = max(all_diff)
		arg_diff = np.argmax(all_diff)
		layers.append(arg_diff)
		#row.append(f"{c_short}-{cur_short}: {max_diff} at {arg_diff}")
		row.append(f"{max_diff:5.2f} at {arg_diff}")

	root_diff = datas[f"{c_short}-rnd"]-datas[f"{c_short}-all"]
	max_diff = max(root_diff)
	arg_diff = np.argmax(root_diff)
	layers.append(arg_diff)
	row.append(f"{max_diff:5.2f} at {arg_diff}")
	rows.append(row)

for row in rows:
	for i in range(len(row)):
		if i != 0: print("& ", end="")
		cell = row[i]
		print(cell + "\t", end="")
	print("\\\\")
#print(rows)

plt.table(cellText=rows, loc='top')
plt.show()

from collections import Counter
print(Counter(layers).most_common())

#plt.plot(xs, nom_all-nom_acc, "o-")
#plt.show()
