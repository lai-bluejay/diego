#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego.test.py was created on 2019/03/20.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import sys
print(sys.path)
import os
os.system('echo $PYTHONPATH')
os.system('echo $DDIR')
print(os.getenv('workspaceFolder'))
print(os.getenv('PYTHONPATH'))
print(os.getenv('DDIR'))
print(os.getenv("PATHPATH"))