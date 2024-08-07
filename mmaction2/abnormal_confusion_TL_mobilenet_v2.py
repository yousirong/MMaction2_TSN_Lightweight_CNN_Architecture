import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
import logging

def my_precision(y_true, y_pred, positive_label=1):
    true_positive  = 0 
    false_positive = 0
    for (i,p) in enumerate(y_pred):
        if p == positive_label and y_true[i] == positive_label: # TP Case
            true_positive += 1
        elif p == positive_label and y_true[i] != positive_label: # FP Case
            false_positive += 1
    return true_positive / (true_positive + false_positive)

def my_recall(y_true, y_pred, positive_label=1):
    true_positive  = 0
    false_negative = 0
    for (i,p) in enumerate(y_pred):
        if p == positive_label and y_true[i] == positive_label: # TP Case
            true_positive += 1
        elif p != positive_label and y_true[i] == positive_label: # FN Case
            false_negative += 1
    return true_positive / (true_positive + false_negative)

def my_f1_score(y_true, y_pred, positive_label=1):
    precision = my_precision(y_true, y_pred, positive_label)
    recall    = my_recall(y_true, y_pred, positive_label)
    return 2.0 / (1/precision + 1/recall)
    
if __name__ == '__main__':
    y_true = []
    y_pred = []

    # f = open("abnormal_logfile_resnet.log", "rt") # 유튜브 실제 데이터 log
    f = open("abnormal_logfile_TL_mobilenet_v2.log", "rt") # TL 테스트 데이터 log
    for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                gt = 'assault' if parts[2].strip() == 'assault' else 'normal'
                predict = 'assault' if parts[3].strip() == 'assault' else 'normal'
                y_true.append(gt)
                y_pred.append(predict)
    f.close()

    # Update label list to reflect the new labeling scheme
    label_list = ['assault', 'normal'] # 0: Assault Detected (abnormal), 1: No Assault Detected (normal)
    
    print(confusion_matrix(y_true, y_pred, labels=label_list))
    print(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['assault', 'normal'])
    print(report)

    print("Precision", metrics.precision_score(y_true, y_pred, average="weighted", labels=label_list))
    print("Recall", metrics.recall_score(y_true, y_pred, average="weighted", labels=label_list))
    print("F1-Score", metrics.f1_score(y_true, y_pred, average="weighted", labels=label_list))