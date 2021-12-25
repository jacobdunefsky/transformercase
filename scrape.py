import requests
from bs4 import BeautifulSoup

lang_name = "Czech"

def get_declension(word):
	t = requests.get("https://en.wiktionary.org/wiki/" + word)
	soup = BeautifulSoup(t.text, 'html.parser')

	lang_header = soup.find("span", id=lang_name, class_="mw-headline").parent

	def is_declension_frame(tag):
		try:
			if tag["class"][0] != "NavFrame": return False
			navhead = tag.div
			if navhead["class"][0] != "NavHead": return False
			if not navhead.text\
				.strip()\
				.lower()\
				.startswith("declension"): return False

			header = tag.find_previous_sibling("h2")

			if header.span["class"][0] != "mw-headline"\
				or header.span["id"] != lang_name:
				return False
			return True
		except Exception as e:
			return False

	declension_frame = lang_header.find_next_sibling(is_declension_frame)
	return declension_frame

# fallback declension tables
# most of the time, when a word doesn't have a provided table, it's
#	an -a or -e feminine noun, so we can use the following surefire
#	rules.

case_to_num = {
	"nominative": 0,
	"genitive": 1,
	"dative": 2,
	"accusative": 3,
	"vocative": 4,
	"locative": 5,
	"instrumental": 6,
}

tables = {
	'a': [
		['a', 'y', 'ě', 'u', 'o', 'ě', 'ou'],
		['y', '', 'ám', 'y', 'y', 'ách', 'ami']
	],
	'e': [
		['e', 'e', 'i', 'i', 'e', 'i', 'í'],
		['e', 'í', 'ím', 'e', 'e', 'ích', 'emi']
	],
}

def get_form(word, case, is_singular, d=None):
	if d is None:
		try:
			d = get_declension(word)
		except Exception as e:
			print(e)
	if not d:
		try:
			return word[:-1] +\
				tables[word[-1]][0 if is_singular else 1][case_to_num[case]]
		except KeyError:
			return None
	def is_match_case(text):
		return text.strip() == case
	row = d.find('th', text=is_match_case)
	cell = row.next_sibling.next_sibling
	if not is_singular: cell = cell.next_sibling.next_sibling
	if not cell.span is None:
		cell = cell.span.contents[0] 
	text = cell.text
	return text.strip()

import pickle
from tqdm import tqdm
def get_cases():
	case_names = list(case_to_num.keys())
	cases = {case_name: [] for case_name in case_names}
	fp = open("czech_nouns.pkl", "rb")

	words = pickle.load(fp)
	for i in tqdm(range(len(words))):
		word = words[i]
		try:
			d = get_declension(word)
		except:
			d = None
		for case_name in case_names:
			form = get_form(word, case_name, True, d=d)
			if form is not None: cases[case_name].append(form)

	ofp = open("cases.pkl", "wb")
	pickle.dump(cases, ofp)
	return cases

