#!/usr/bin/python

import os

while True:

    try:
        os.system("ab -n 5000 -c 10 http://localhost:8501/train")

    except Exception as e:
        print(e)
