import pickle
import numpy as np

words_by_case = pickle.load(open("cases.pkl", "rb"))
words_dict = {}
for case in words_by_case:
	for word in words_by_case[case]:
		if not word in words_dict:
			words_dict[word] = [case]
		else:
			words_dict[word].append(case)

for word in words_dict:
	new_list = []
	for case in words_by_case:
		if case in words_dict[word]:
			new_list.append(1)
		else:
			new_list.append(0)
	words_dict[word] = np.array(new_list)	

new_dict = {"word": [], "cases": []}

for word in words_dict:
	new_dict['word'].append(word)
	new_dict['cases'].append(words_dict[word])

import pandas as pd
words_df = pd.DataFrame(data=new_dict)
pickle.dump(words_df, open("words_df.pkl", "wb"))
