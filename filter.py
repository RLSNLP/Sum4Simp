import math
import scipy.stats

file1 = open(r'train.src', 'r') # The sentence simplification training set (WikiLarge, e.g.)
file2 = open(r'train.dst', 'r')

store1 = []
store2 = []
lines = file1.readlines()
for line in lines:
    store1.append(line.strip().lower())
lines = file2.readlines()
for line in lines:
    store2.append(line.strip().lower())

def normal_distribution_function(x, mean, std):
    if x <= mean:
        return 1
    else:
        result = 2 * (1 - scipy.stats.norm(mean, std).cdf(x))
        return round(result, 3)

def for_sari_normal_distribution_function(x, mean, std):
    if x >= mean:
        return 1
    else:
        result = 2 * scipy.stats.norm(mean, std).cdf(x)
        return round(result, 3)

# word/sentence
import nltk
import numpy as np
length_ratio = []
for idx, line in enumerate(store1):
    tmp_src = len(nltk.tokenize.word_tokenize(line))/len(nltk.tokenize.sent_tokenize(line))
    tmp_dst = len(nltk.tokenize.word_tokenize(store2[idx]))/len(nltk.tokenize.sent_tokenize(store2[idx]))
    length_ratio.append(tmp_src/tmp_dst)
length_ratio_mean = np.mean(length_ratio)
length_ratio_std = np.std(length_ratio,ddof=1)
# print("")
# print(length_ratio_mean)
# print(length_ratio_std)
# print("")

# avg_word_complexity
file3 = open(r'lexicon.tsv', 'r')
lines = file3.readlines()
lexicon = {}
for line in lines:
    lexicon[line.split()[0].lower()] = float(line.split()[1])
complexity_disperion = []
for idx, line in enumerate(store1):
    tmp_src = nltk.tokenize.word_tokenize(line)
    tmp_dst = nltk.tokenize.word_tokenize(store2[idx])
    tmp_src_complexity = 0
    tmp_dst_complexity = 0
    tmp_src_length = 0
    tmp_dst_length = 0
    for word in tmp_src:
        if lexicon.get(word) != None:
            tmp_src_complexity = tmp_src_complexity + lexicon.get(word)
            tmp_src_length += 1
    for word in tmp_dst:
        if lexicon.get(word) != None:
            tmp_dst_complexity = tmp_dst_complexity + lexicon.get(word)
            tmp_dst_length += 1
    if tmp_src_length == 0:
        tmp_src_length += 1
    if tmp_dst_length == 0:
        tmp_dst_length += 1
    # print(tmp_src_complexity/tmp_src_length)
    # print(tmp_dst_complexity/tmp_dst_length)
    # if idx == 4:
        # break
    complexity_disperion.append((tmp_dst_complexity/tmp_dst_length)-(tmp_src_complexity/tmp_src_length))
complex_mean = np.mean(complexity_disperion)
complex_std = np.std(complexity_disperion,ddof=1)
# print("")
# print(complex_mean)
# print(complex_std)
# print("")

# odds_ratio
file4 = open(r'dict.txt', 'r')
ratio_dict = {}
lines = file4.readlines()
for line in lines:
    line = line.split()
    ratio_dict[line[0]] = float(line[1])
file5 = open(r'valid.src', 'r')
file6 = open(r'valid.dst', 'r')
src_lines = file5.readlines()
dst_lines = file6.readlines()
store3 = []
store4 = []
for line in src_lines:
    store3.append(line.strip().lower())
for line in dst_lines:
    store4.append(line.strip().lower())
odds_ratio = []
for idx, line in enumerate(store3):
    tmp_src = nltk.tokenize.word_tokenize(line)
    tmp_dst = nltk.tokenize.word_tokenize(store4[idx])
    # print(tmp_src)
    # print(tmp_dst)
    src_odds_ratio = 0
    dst_odds_ratio = 0
    src_count = 0
    dst_count = 0
    for word in tmp_src:
        if ratio_dict.get(word) != None:
            src_odds_ratio += ratio_dict.get(word)
            src_count += 1
    for word in tmp_dst:
        if ratio_dict.get(word) != None:
            dst_odds_ratio += ratio_dict.get(word)
            dst_count += 1
    if dst_count == 0:
        dst_count = 1
    if src_count == 0:
        src_count = 1
    odds_ratio.append(dst_odds_ratio/dst_count - src_odds_ratio/src_count)
odds_ratio_mean = np.mean(odds_ratio)
odds_ratio_std = np.std(odds_ratio,ddof=1)
# print("")
# print(odds_ratio_mean)
# print(odds_ratio_std)
# print("")

file1.close()
file2.close()
file3.close()
file4.close()
file5.close()
file6.close()


# Here is the results for WikiLarge, you can use them directly. The above codes can be commented out.
length_ratio_mean = 2.90
length_ratio_std = 5.78

complex_mean = -0.03
complex_std = 0.43

odds_ratio_mean = -0.19
odds_ratio_std = 1.16

SARI_mean = 37.61
SARI_std = 23.50

# Select suitable pairs

file1 = open(r'complex.txt', 'r') # Aligned sentence pairs
file2 = open(r'simple.txt', 'r')

store1 = []
store2 = []

lines = file1.readlines()
for line in lines:
    store1.append(line.strip().lower())

lines = file2.readlines()
for line in lines:
    store2.append(line.strip().lower())

import nltk
import numpy as np
file3 = open(r'lexicon.tsv', 'r')
lines = file3.readlines()
lexicon = {}
for line in lines:
    lexicon[line.split()[0].lower()] = float(line.split()[1])

file4 = open(r'dict.txt', 'r')
ratio_dict = {}
lines = file4.readlines()
for line in lines:
    line = line.split()
    ratio_dict[line[0]] = float(line[1])

file5 = open(r'sari_value.txt', 'r')
sari_values = []
lines = file5.readlines()
for line in lines:
    sari_values.append(float(line.strip()))

total_score = []

for idx, line in enumerate(store1):
    if idx % 50000 == 0:
        print(idx)
    if len(nltk.tokenize.word_tokenize(line)) == 0 or len(nltk.tokenize.word_tokenize(store2[idx])) == 0:
        total_score.append(0)
        continue
    # length_ratio
    tmp_src = len(nltk.tokenize.word_tokenize(line))/len(nltk.tokenize.sent_tokenize(line))
    tmp_dst = len(nltk.tokenize.word_tokenize(store2[idx]))/len(nltk.tokenize.sent_tokenize(store2[idx]))
    tmp_length_ratio = tmp_src/tmp_dst
    length_ratio_score = normal_distribution_function(tmp_length_ratio, length_ratio_mean, length_ratio_std)
    # complex
    tmp_src = nltk.tokenize.word_tokenize(line)
    tmp_dst = nltk.tokenize.word_tokenize(store2[idx])
    tmp_src_complexity = 0
    tmp_dst_complexity = 0
    tmp_src_length = 0
    tmp_dst_length = 0
    for word in tmp_src:
        if lexicon.get(word) != None:
            tmp_src_complexity = tmp_src_complexity + lexicon.get(word)
            tmp_src_length += 1
    for word in tmp_dst:
        if lexicon.get(word) != None:
            tmp_dst_complexity = tmp_dst_complexity + lexicon.get(word)
            tmp_dst_length += 1
    if tmp_src_length == 0:
        tmp_src_length += 1
    if tmp_dst_length == 0:
        tmp_dst_length += 1
    tmp_complexity = tmp_dst_complexity/tmp_dst_length - tmp_src_complexity/tmp_src_length
    complexity_score = normal_distribution_function(tmp_complexity, complex_mean, complex_std)
    # odds_ratio
    tmp_src = nltk.tokenize.word_tokenize(line)
    tmp_dst = nltk.tokenize.word_tokenize(store2[idx])
    src_odds_ratio = 0
    dst_odds_ratio = 0
    src_count = 0
    dst_count = 0
    for word in tmp_src:
        if ratio_dict.get(word) != None:
            src_odds_ratio += ratio_dict.get(word)
            src_count += 1
    for word in tmp_dst:
        if ratio_dict.get(word) != None:
            dst_odds_ratio += ratio_dict.get(word)
            dst_count += 1
    if dst_count == 0:
        dst_count = 1
    if src_count == 0:
        src_count = 1
    tmp_odds_ratio = dst_odds_ratio/dst_count - src_odds_ratio/src_count
    odds_ratio_score = normal_distribution_function(tmp_odds_ratio, odds_ratio_mean, odds_ratio_std)
    # SARI
    sari_value = sari_values[idx]
    sari_score = for_sari_normal_distribution_function(sari_value, SARI_mean, SARI_std)
    # Total score
    total_score.append(length_ratio_score + complexity_score + odds_ratio_score + sari_score)

file6 = open(r'total_score.txt', 'w')
for score in total_score:
    file6.write(str(score))
    file6.write('\n')

file1.close()
file2.close()
file3.close()
file4.close()
file5.close()
file6.close()
