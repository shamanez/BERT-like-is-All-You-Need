import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

data_mos=pd.read_csv('mos.csv')

true_mos=data_mos['True']
pred_mos=data_mos['Predicted']


 

f1_score_mos=f1_score(true_mos, pred_mos, average='weighted')

print("F1 score mos",f1_score_mos)
