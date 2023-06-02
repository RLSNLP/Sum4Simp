import os
import nltk
import copy
import random
import time
random.seed(1)

filenames = []
cnn_filenames = os.listdir(r'./cnn_stories_tokenized/')
for cnn_name in cnn_filenames:
    filenames.append(r'./cnn_stories_tokenized/'+cnn_name)
dm_filenames = os.listdir(r'./dm_stories_tokenized/')
for dm_name in dm_filenames:
    filenames.append(r'./dm_stories_tokenized/'+dm_name)
random.shuffle(filenames)
print("Total files:")
print(len(filenames))
print("")

file1 = open(r'complex.txt', 'w')
file2 = open(r'simple.txt', 'w')

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
model.max_seq_length = 128

start = time.time()

num_bigger_than_0_8 = 0
num_smaller_than_0_6 = 0
other = 0
total_src_sentences = 0
total_tgt_sentences = 0

def takeSecond(elem):
    return elem[1]

for idx, name in enumerate(filenames):
    if idx % 100000 == 0:
        print(idx)
    file = open(name, 'r')
    src = []
    tgt = []
    lines = file.readlines()
    signal = False
    for line in lines:
        if line.strip() == "":
            continue
        if line.strip() == "@highlight":
            signal = True
            continue
        if signal:
            tgt.append(line.strip())
            signal = False
            continue
        src_sentences = nltk.sent_tokenize(line.strip())
        for src_sentence in src_sentences:
            src.append(src_sentence)

    total_src_sentences += len(src)
    total_tgt_sentences += len(tgt)
    file.close()

    if len(src) == 0 or len(tgt) == 0:
        continue

    embeddings1 = model.encode(src, convert_to_tensor=True)
    embeddings2 = model.encode(tgt, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)


    for j in range(len(tgt)):
        str_src = ""
        str_tgt = tgt[j]

        tmp_dict = []
        for i in range(len(src)):
            tmp_dict.append([src[i], cosine_scores[i][j].item(), i])
        tmp_dict.sort(key=takeSecond, reverse=True)

        if tmp_dict[0][1] >= 0.8:
            str_src = str_src + tmp_dict[0][0]
            num_bigger_than_0_8 += 1
        elif tmp_dict[0][1] < 0.6:
            num_smaller_than_0_6 += 1
            continue
        else:
            other += 1
            str_src_list = [tmp_dict[0][2]]
            tmp_str_src_list = [tmp_dict[0][2]]
            for i in range(1, len(tmp_dict)):
                tmp_str_src_list.append(tmp_dict[i][2])
                tmp_str_src_list.sort()
                tmp_str_src = ""
                for element in tmp_str_src_list:
                    tmp_str_src = tmp_str_src + src[element] + " "
                # print(tmp_str_src)
                tmp_embeddings1 = model.encode(tmp_str_src, convert_to_tensor=True)
                tmp_embeddings2 = model.encode(str_tgt, convert_to_tensor=True)
                tmp_cosine_scores = util.pytorch_cos_sim(tmp_embeddings1, tmp_embeddings2)
                # print(tmp_cosine_scores[0][0].item())
                # print("")
                if tmp_cosine_scores[0][0].item() < 0.7 or len(str_src_list) >= 2:
                    # print("result:")
                    for element in str_src_list:
                        str_src = str_src + src[element] + " "
                    # print(str_src)
                    break
                else:
                    str_src_list = copy.deepcopy(tmp_str_src_list)

        print(str_src.strip())
        print(str_tgt.strip())

        file1.write(str_src.strip())
        file1.write("\n")
        file2.write(str_tgt.strip())
        file2.write("\n")

    # for j in range(len(tgt)):
    #     signal = False
    #     str_src = ""
    #     str_tgt = tgt[j]
    #     for i in range(len(src)):
    #         if cosine_scores[i][j] >= 0.6:
    #             signal = True
    #             str_src = str_src + src[i] + " "
    #     #         print("")
    #     #         print(src[i])
    #     #         print(tgt[j])
    #     #         print(cosine_scores[i][j])
    #     # print("")
    #     if not signal:
    #         continue
    #     file1.write(str_src)
    #     file1.write("\n")
    #     file2.write(str_tgt)
    #     file2.write("\n")

end = time.time()
print("CPU time")
print(end-start)

print("total_src_sentences")
print(total_src_sentences)
print("total_tgt_sentences")
print(total_tgt_sentences)

file1.close()
file2.close()