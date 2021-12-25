import fasttext

if not 'model' in globals():
	model = fasttext.load_model("/media/jacob/DATA1/T0RR3NTZ/cc.cs.300.bin")
else:
	model = globals()['model']

import pickle
words_by_case = pickle.load(open("cases.pkl", "rb"))

import numpy as np
def get_nearest_word(vec, ignore=None):
	min_dist = -1
	min_word = None
	for word in model.words:
		if ignore is not None:
			if word == ignore: continue
		dist = np.linalg.norm(vec-model[word])	
		if min_dist == -1 or dist < min_dist:
			min_dist = dist
			min_word = word
	return min_word	
		

from tqdm import tqdm
def get_case_diff_mean(c1, c2):
	assert(len(c1)==len(c2))
	s=0
	for i in range(len(c1)):
		c = model[c1[i]]-model[c2[i]]
		s += c/np.linalg.norm(c)
	return s/len(c1)

import numpy as np

def get_case_angle_var(c1, c2):
	m = get_case_diff_mean(c1, c2)
	magnitude = np.sqrt(np.dot(m,m))
	mm = np.average(np.square(m))
	v = 0
	ignored = 0 #num ignored samples due to too small size
	#cosines = []
	angles = []
	for i in tqdm(range(len(c1))):
		c = model[c1[i]]-model[c2[i]]
		# cosine similarity, but checking to see if we have div by 0
		# if div by 0, then ignore sample
		cc = np.average(np.square(c))
		denom = np.sqrt(cc*mm)
		if denom == 0:
			continue
		cm = np.average(c*m)
		#cosines.append(cm/denom)
		angles.append(math.acos(cm/denom))
	#sines = [math.sin(math.acos(c)) for c in cosines]
	#vectors = np.array(list(zip(cosines,sines)))
	#vecmean = vectors.mean(axis=0)
	#avgangle = math.atan2(vecmean[1], vecmean[0])
	avgangle = sum(angles)/len(angles)
	return avgangle*180/math.pi

import pickle
words_by_case = pickle.load(open("cases.pkl", "rb"))

case_to_num = {
	"nominative": 0,
	"genitive": 1,
	"dative": 2,
	"accusative": 3,
	"vocative": 4,
	"locative": 5,
	"instrumental": 6,
}
num_to_case = {num: case for case, num in case_to_num.items()}
#non_nom = []
limit = 1000

def construct_control(exclude):
	non_case = []
	for i in range(limit):
		case = num_to_case[(
			(i%(len(case_to_num)-1))+case_to_num[exclude]+1	
		)%len(case_to_num)]
		non_case.append(words_by_case[case][i])
	return non_case
"""for i in range(limit):
	case = num_to_case[(i%6)+1]
	non_nom.append(words_by_case[case][i])"""

import itertools
import random
all_words = list(itertools.chain.from_iterable(words_by_case.values()))
random.shuffle(all_words)

import math

def get_angle(a,b):
	c = get_case_cos_var(a,b)
	return 180*math.acos(c)/math.pi

import os

def run_test(case_name, other_case):
	print(f"{case_name}, {other_case}")
	aav = get_case_angle_var(words_by_case[case_name][:limit],
		words_by_case[other_case][:limit])
	print("\tAngle variance: " +\
		str(aav)
	)
	return aav

def run_root(case_name):
	print(f"{case_name}, same root")
	non_case = construct_control(case_name)
	av = get_case_angle_var(words_by_case[case_name][:limit],
		non_case)
	print("\tAngle variance: " +\
		str(av)
	)
	return av

def run_random(case_name):
	print(f"{case_name}, random")
	non_case = all_words[:limit]
	av = get_case_angle_var(words_by_case[case_name][:limit],
		non_case)
	print("\tAngle variance: " +\
		str(av)
	)
	return av

def do_it():
	data = {}
	for i in range(len(words_by_case)):
		c1 = list(words_by_case.keys())[i]
		c1_short = c1[:3]
		data[f"{c1_short}-all"] = run_root(c1)
		data[f"{c1_short}-rnd"] = run_random(c1)
		for j in range(i):
			c2 = list(words_by_case.keys())[j]
			c2_short = c2[:3]
			data[f"{c1_short}-{c2_short}"] = x = run_test(c1, c2)
			print(f"{c1_short}-{c2_short}: {x}")

	pickle.dump(data, open("ft-data-alt.pkl", "wb") )

do_it()
#if __name__ == "__main__":
#	run_test("vocative", "genitive")
