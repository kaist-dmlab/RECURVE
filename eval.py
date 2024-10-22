import numpy as np
from sklearn import metrics
from scipy import signal


def AUC(gt, values):
    '''
    gt: ground truth binary boundary label with margin where 1 is gradual and 2 is abrupt boundary
    values: change point metric ranging from 0 to 1 so that we can threshold for auc
    '''
    gt_total = np.array(gt>=1, dtype=np.int32)
    fpr, tpr, thresholds = metrics.roc_curve(gt_total, values, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def LOC(gtp, pred):
    '''
    Average of minimum distance between predicted boundary and true boundaries.
    gtp: ground truth binary boundary label without margin where 1 is gradual and 2 is abrupt boundary
    pred: predicted binary boundary label
    '''
    true_indice = np.where(gtp[1:]!=gtp[:-1])[0].reshape(-1,1)
    pred_indice = np.where(pred==1)[0].reshape(-1,1)
    indice, distance = metrics.pairwise_distances_argmin_min(pred_indice, true_indice)
    return np.average(distance)

def pred_by_finding_peaks(measures):
    boundary_index, _ = signal.find_peaks(measures)
    boundary_pred = np.zeros_like(measures)
    boundary_pred[boundary_index] = 1
    return boundary_pred

def best_f1_threshold(boundary_labels, values):
    precision, recall, thresholds = metrics.precision_recall_curve(boundary_labels>=1, values)
    f1_scores = 2*recall*precision/(recall+precision)
    f1_scores[np.isnan(f1_scores)] = 0 # make 0 division as value 0
    # print('Best threshold: ', thresholds[np.argmax(f1_scores)])
    # print('Best F1-Score: ', np.max(f1_scores))
    return thresholds[np.argmax(f1_scores)]