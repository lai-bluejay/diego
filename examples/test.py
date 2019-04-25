#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/examples.test.py was created on 2019/03/20.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import sys
print(sys.path)
from diego import classifier

x = classifier.DiegoClassifier()
print(x)

            
import operator  
phrase_idf=idf(output_list)
phrase_idf_sort=sorted(phrase_idf.items(),key=operator.itemgetter(1))
idf_list=[]
with open(r"C:\Users\FFZX-liuyy\Desktop\AutoPhrase\models\New folder1\controlled_terms\segmentation_phrase_idf.txt","w") as f_idf:
    for parase_idf in phrase_idf_sort:
        for item in parase_idf:
            idf_list.append(item)
        idf_list = [str(idf) for idf in idf_list]
        line = ":".join(idf_list)
    f_idf.write('\n'.join(idf_list))  
    print("phrases idf end")