import transformers
import torch

model_str = "ufal/robeczech-base"

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

tokenizer = AutoTokenizer.from_pretrained(model_str)
config = AutoConfig.from_pretrained(model_str, output_hidden_states=True)
model = AutoModelForMaskedLM.from_pretrained(model_str, config=config).to("cuda:0")

def get_embedding(s, l):
	tokens = tokenizer(s, return_tensors="pt").to("cuda:0")
	#TODO: is final layer -1 or 0?
	final_layer = model(**tokens).hidden_states[l]
	# supposedly, mean works better than [CLS] embedding
	embedding = final_layer.mean(dim=1)[0]

	# actually, try [SEP] embedding
	#embedding = final_layer[:,-1][0]
	return embedding.detach().cpu().numpy()

from tqdm import tqdm
def get_case_diff_mean(c1, c2,l):
	assert(len(c1)==len(c2))
	s=0
	for i in range(len(c1)):
		c = get_embedding(c1[i],l)-get_embedding(c2[i],l)
		s += c
	return s/len(c1)

import numpy as np

def get_case_angle_var(c1, c2,l):
	m = get_case_diff_mean(c1, c2,l)
	magnitude = np.sqrt(np.dot(m,m))
	mm = np.average(np.square(m))
	v = 0
	ignored = 0 #num ignored samples due to too small size
	#cosines = []
	angles = []
	for i in tqdm(range(len(c1))):
		c = get_embedding(c1[i],l)-get_embedding(c2[i],l)
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
limit = 400

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

def run_random(case_name):
	c1_short = case_name[:3]
	rel_str = c1_short + "-random"

	aav = []

	print(rel_str)
	for l in tqdm(range(13)):
		print("Layer " + str(l) + ": ")
		av = get_case_angle_var(words_by_case[case_name][:limit],
			all_words[:limit],l)
		print("\tAngle variance: " +\
			str(av)
		)
		aav.append(av)

	pickle.dump(aav, open(rel_str + ".pkl", "wb"))

def run_test(case_name, other_case):
	non_case = construct_control(case_name)

	c1_short = case_name[:3]
	c2_short = other_case[:3]
	rel_str = c1_short + "-" + c2_short

	aav = []

	print(rel_str)
	for l in tqdm(range(13)):
		print("Layer " + str(l) + ": ")
		av = get_case_angle_var(words_by_case[case_name][:limit],
			words_by_case[other_case][:limit],l)
		print("\tAngle variance: " +\
			str(av)
		)
		aav.append(av)

	pickle.dump(aav, open(rel_str + ".pkl", "wb"))

	if os.path.exists(c1_short + "-all.pkl"): return

	print("=====")

	rav = []
	print(c1_short + "-all")
	for l in range(13):
		print("Layer " + str(l) + ": ")
		av = get_case_angle_var(words_by_case[case_name][:limit],
			non_case,l)
		print("\tAngle variance: " +\
			str(av)
		)
		rav.append(av)

	pickle.dump(rav, open(c1_short + "-all.pkl", "wb"))

for case in words_by_case:
	run_random(case)
