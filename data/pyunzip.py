#!/usr/bin/env python3
import sys
import nltk
from zipfile import PyZipFile
for zip_file in sys.argv[1:]:
    pzf = PyZipFile(zip_file)
    pzf.extractall()

nltk.download('omw-1.4')