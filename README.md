# Sum4Simp
Codes and the dataset for the paper "Exploiting Summarization Data to Help Text Simplification"(https://aclanthology.org/2023.eacl-main.3.pdf)

The S4S dataset is a stardard sentence simplification dataset mentioned in the paper. You could also mix them all for data augmentation.

If you want to obtain the aligned sentence pairs yourself, you should download the CNN and DM datasets at first. Then, you need to run 'python align.py'.

If you want to filter the suitable sentence pairs from the aligned pairs, you should calculate the attribute values at first. We have upload some example files (for WikiLarge) and you could run 'python filter.py' and check out the total scores. You could set a threshold to filter the pairs you need.

If you have any questions, please contact us: sunrenliangpku@gmail.com
