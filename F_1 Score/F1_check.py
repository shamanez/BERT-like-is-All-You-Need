import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

data_sad=pd.read_csv('sad.csv')
data_ang=pd.read_csv('ang.csv')
data_neu=pd.read_csv('neu.csv')
data_hap=pd.read_csv('hap.csv')

true_sad=data_sad['True']
pred_sad=data_sad['Predicted']

true_ang=data_ang['True']
pred_ang=data_ang['Predicted']

true_hap=data_hap['True']
pred_hap=data_hap['Predicted']

true_neu=data_neu['True']
pred_neu=data_neu['Predicted']

 

f1_score_sad=f1_score(true_sad, pred_sad, average='weighted')
f1_score_ang=f1_score(true_ang, pred_ang, average='weighted')
f1_score_neu=f1_score(true_neu, pred_neu, average='weighted')
f1_score_hap=f1_score(true_hap, pred_hap, average='weighted')

print("F1 score sad",f1_score_sad)
print("F1 score ang",f1_score_ang)
print("F1 score hap",f1_score_hap)
print("F1 score neu",f1_score_neu)