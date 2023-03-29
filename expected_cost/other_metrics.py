import numpy as np
from sklearn.metrics import roc_curve 
from expected_cost import ec, utils


#########################################################################################
# Definition of a few standard metrics computed from the confusion matrix

def Fscore(K10, K01, N0, N1):
    K11 = N1-K10
    Recall    = K11/N1
    Precision = K11/(K11+K01) if K11+K01>0 else 0
    # Returns Fscore
    return 2 * Precision*Recall/(Recall+Precision) if K11>0 else 0
    

def MCCoeff(K10, K01, N0, N1):
    K11 = N1-K10
    K00 = N0-K01
    num = K00 * K11 - K01 * K10
    den = np.sqrt(N0 * N1 * (K01+K11) * (K10 + K00))
    return num/den if den>0 else (np.inf if num>0 else -np.inf)


def LRplus(K10, K01, N0, N1):
    R10 = K10 / N1
    R01 = K01 / N0
    return (1-R10)/R01 if R01>0 else np.inf


def Accuracy(K00, K11, K):
    return (K00+K11)/K


def BalAccuracy(C, K00, K11, N0, N1):
    return 1/C*(K00/N0+K11/N1)


def NetBenefit():
    pass



#######################################################################
# Other metrics computed using EC so that is obvious that they are
# specific cases of EC. 

def acc_from_EC(targets, decisions, costs=None, priors=None, sample_weight=None, adjusted=False):
    '''
    Accuracy is 1-Error rate
    Error rate is the EC when using:
    - The priors in the evaluation data
    - The 0-1 cost matrix 
    '''
    EC = ec.average_cost(targets, decisions, costs, priors, adjusted=adjusted)
    return 1-EC


def bal_acc_from_EC(targets, decisions, costs=None, priors=None, sample_weight=None, adjusted=False):
    '''
    Balanced Accuracy is defined as the average of the recall values 
    (the fraction of samples from a certain class that are labelled correctly) 
    over all classes. It is defined as 1-BalErrorRate where BalErrorRate is the
    Error rate. In terms of the EC the BalErrorRate is the EC when using: 
    - cij = 1/(CP_i) in the cost matrix
    '''
    
    EC = ec.average_cost(targets, decisions, costs, priors, adjusted=adjusted)
    
    return 1-EC

def net_benefit_from_EC(targets, decisions, pt=None, priors=None, sample_weight=None, adjusted=False):
    '''
    Net benefit can be defined as a function of EC
    P_2 - min(P_2, (p_t/1-p_t) P_1) NEC
    '''
    c12 = pt/(1-pt)
    costs = ec.cost_matrix([[0, c12],[1,0]])
    net_factor = priors[1] - min(priors[1], (pt/1-pt)*priors[0])
    NEC = ec.average_cost(targets, decisions, costs, priors, adjusted=adjusted)
    return net_factor * NEC

def one_minus_f_beta_from_EC(targets, scores, thr, priors=None, beta=None):
    
    # Number of samples from each class
    N0 = sum(targets==0)
    N1 = sum(targets==1)
    K = N0 + N1
    # Number of samples of class 0 with a score larger than the thr (ie, labelled as class 1)
    K01 = np.sum(scores[targets==0]>thr)
    # Number of samples of class 1 with a score smaller than the thr (ie, labelled as class 0)
    K10 = np.sum(scores[targets==1]<thr)
    P1 = priors[0]
    P2 = priors[1]
    
    costs_beta = ec.cost_matrix([[0, np.sqrt(beta)],[1,0]])
    R = utils.compute_R_matrix_from_counts_for_binary_classif(K01, K10, N0, N1)
    R_ast_2 = R[0][1]+R[1][1]
    
    NEC_beta = ec.average_cost_from_confusion_matrix(R, priors, costs_beta, adjusted=True)
    
    return min(np.sqrt(beta)*P2, P1)*(NEC_beta/(np.sqrt(beta)*P2+R_ast_2))

def mccoeff_from_EC():
    pass

def lrplus_from_EC():
    pass





#######################################################################
# Other ways of computing minDCF

# MinDCF as computed in Pron papers
def min_cost(labels, scores, cost_thr=None, cost_fp=0.5):
    # sourcery skip: extract-method, lift-return-into-if, move-assign
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1-tpr
    
    # Use the best cheating threshold to get the min cost
    cost_normalizer = min(cost_fp, 1.0)
    cost = (cost_fp * fpr + fnr)/cost_normalizer
    min_cost_idx = np.argmin(cost)
    min_cost_thr = thr[min_cost_idx]
    min_cost     = cost[min_cost_idx]
    min_cost_fpr = fpr[min_cost_idx]
    min_cost_fnr = fnr[min_cost_idx]
    
    
    if cost_thr is not None:
        det_pos = labels[scores>cost_thr]
        det_neg = labels[scores<=cost_thr]
        act_cost_fpr = np.sum(det_pos==0)/np.sum(labels==0)
        act_cost_fnr = np.sum(det_neg==1)/np.sum(labels==1)
        act_cost = (cost_fp * act_cost_fpr + act_cost_fnr)/cost_normalizer
    else:
        act_cost_fpr = min_cost_fpr
        act_cost_fnr = min_cost_fnr
        act_cost = min_cost
    
    return act_cost
    


## MinDCF from PyBOSARIS
def compute_labels_predicted(scores, threshold=0):
    # Compute labels predicted for an specific threhold
    return [1 if i>threshold else 0 for i in scores]  

def cost_det(th, scores, labels, Ptar=0.5, cost_miss=1, cost_fa=1):
    
    labels_predicted = compute_labels_predicted(scores, threshold=th)
    miss = [1 if i>j else 0 for i,j in zip(labels, labels_predicted)]
    Pmiss = sum(np.array(miss)==1)/len(miss)

    fa = [1 if i<j else 0 for i,j in zip(labels, labels_predicted)]
    Pfa = sum(np.array(fa)==1)/len(fa)

    cdet = cost_miss * Ptar * Pmiss + cost_fa * (1-Ptar) * Pfa

    cdef = np.min([cost_miss*Ptar, cost_fa*(1-Ptar)])

    return cdet/cdef, Pmiss, Pfa