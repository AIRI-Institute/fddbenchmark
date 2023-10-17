import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment


class FDDEvaluator():
    def __init__(self, step_size: int):
        self.step_size = step_size
        
    def evaluate(self, labels, pred):
        # labels should be non-negative integer values, normal is 0
        assert np.all(np.sort(np.unique(labels)) == np.arange(labels.max() + 1))
        # confustion matrix: rows are truth classes, columns are predicted classes
        fdd_cm = confusion_matrix(labels, pred, labels=np.arange(labels.max() + 1))
        metrics = {'detection': dict(), 'diagnosis': dict(), 'clustering': dict(), 'classification': dict()}
        metrics['confusion_matrix'] = fdd_cm
        metrics['detection']['TPR'] = fdd_cm[1:, 1:].sum() / fdd_cm[1:, :].sum()
        metrics['detection']['FPR'] = fdd_cm[0, 1:].sum() / fdd_cm[0, :].sum()

        real_change_point = labels[labels != 0].reset_index().groupby(['run_id'])['sample'].min()
        pred_change_point = pred[pred != 0].reset_index().set_index(['run_id'])['sample']
        pred_real_change_point = pd.merge(
            pred_change_point, 
            real_change_point, 
            how='left', 
            on='run_id', 
            suffixes=('', '_real')
        )
        valid_change_point = pred_real_change_point['sample_real'] <= pred_real_change_point['sample']
        pred_change_point = pred_change_point[valid_change_point].groupby(['run_id']).min()
        detection_delay = (pred_change_point - real_change_point) * self.step_size
        metrics['detection']['ADD'] = detection_delay.mean()
        
        correct_diagnosis = fdd_cm[1:, 1:].diagonal()
        tp = fdd_cm[1:, 1:].sum()
        metrics['diagnosis']['CDR_total'] = correct_diagnosis.sum() / tp
        
        correct_classification = fdd_cm.diagonal()
        metrics['classification']['TPR'] = correct_classification / fdd_cm.sum(axis=1)
        metrics['classification']['FPR'] = fdd_cm[0] / fdd_cm[0].sum()
        
        metrics['clustering']['ACC'] = cluster_acc(labels.values, pred.values)
        metrics['clustering']['NMI'] = normalized_mutual_info_score(labels.values, pred.values)
        metrics['clustering']['ARI'] = adjusted_rand_score(labels.values, pred.values)
        return metrics
    
    def str_metrics(self, labels, pred):
        metrics = self.evaluate(labels, pred)
        str_metrics = []
        str_metrics.append('FDD metrics\n-----------------')
        
        str_metrics.append('TPR/FPR:')
        for i in np.arange(labels.max()).astype('int'):
            str_metrics.append('    Fault {:02d}: {:.4f}/{:.4f}'.format(i+1, metrics['classification']['TPR'][i+1], metrics['classification']['FPR'][i+1]))

        str_metrics.append('Detection TPR: {:.4f}'.format(metrics['detection']['TPR']))
        str_metrics.append('Detection FPR: {:.4f}'.format(metrics['detection']['FPR']))
        str_metrics.append('Average Detection Delay (ADD): {:.2f}'.format(metrics['detection']['ADD']))
        str_metrics.append('Total Correct Diagnosis Rate (Total CDR): {:.4f}'.format(metrics['diagnosis']['CDR_total']))

        str_metrics.append('\nClustering metrics\n-----------------')
        str_metrics.append('Adjusted Rand Index (ARI): {:.4f}'.format(metrics['clustering']['ARI']))
        str_metrics.append('Normalized Mutual Information (NMI): {:.4f}'.format(metrics['clustering']['NMI']))
        str_metrics.append('Unsupervised Clustering Accuracy (ACC): {:.4f}'.format(metrics['clustering']['ACC']))
        return '\n'.join(str_metrics)

    def print_metrics(self, labels, pred):
        print(self.str_metrics(labels, pred))


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed. 
    Taken from https://github.com/XifengGuo/DEC-DA
    # Arguments
        y: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    
    return sum([w[ind[0][i], ind[1][i]] for i in range(ind[0].size)]) * 1.0 / y_pred.size
