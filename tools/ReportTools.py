import os

import pandas as pd
from sklearn.metrics import  confusion_matrix, classification_report, auc, roc_auc_score, \
    average_precision_score




def reports(y, p):
    cm = confusion_matrix(y, p)
    cr = classification_report(y, p, digits=5, zero_division=0)
    roc_score = roc_auc_score(y, p)
    ap_score = average_precision_score(y, p)
    print(cm)
    print(cr)
    print('auc:%.5f' % roc_score)
    print('ap:%.5f' % ap_score)
    cr_dict = classification_report(y, p, digits=5, zero_division=0,output_dict=True)
    return pd.Series(cr_dict['macro avg']).drop('support')

# def get_report(cancer, model: str, args: dict, loop_time, seq, _print=True):
#     cr1 = classification_report(y1, p1, digits=5, zero_division=0)
#     cr2 = classification_report(y2, p2, digits=5, zero_division=0)
#     if _print:
#         print(cm)
#         print('auc:%.4f' % _auc)
#         print(" Model Prediction Report ")
#         print(cr)
#     data = [[model, args,
#              fpr, tpr, _auc, str(cm), str(cr), gridID]]
#     df = pd.DataFrame(data=data, columns=['model_type', 'args', 'fpr', 'tpr', 'auc', 'confusion_matrix'
#         , 'classification_report'])
#     import time
#     lpg = LogPathGetter()
#     path = lpg.get_path('result_' + time.strftime("%Y_%m_%d", time.localtime()) + '.csv')
#     log_path = 'log/
#     df.to_csv('log/result_' + time.strftime("%Y_%m_%d", time.localtime()) + '.csv', mode='a',
#               header=not os.path.exists(log_path), index=False)
