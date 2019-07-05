import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.system(" /opt/spark/bin/spark-submit /data/lin/code/code_git/RecommendOnSpark/recommend_train.py")

