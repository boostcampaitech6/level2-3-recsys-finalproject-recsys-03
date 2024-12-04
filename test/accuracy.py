from sklearn.metrics import recall_score
import pandas as pd
import numpy as np

results = pd.read_csv('service_test.csv')
recall = []

for row in results.iterrows():
    ans = [elem.strip(" '") for elem in row[1]['ans'].strip("[]").split(',')]
    rec = [elem.strip(" '") for elem in row[1]['recommended'].strip("[]").split(',')]
    correct = sum(1 for id in rec if id in ans)
    recall.append(correct/20)

results['recall'] = recall
results.to_csv('service_test.csv')
print(np.mean(recall))
#1차 배포 : 0.0023985239852398524
#30개 이상: 0.0020833333333333333