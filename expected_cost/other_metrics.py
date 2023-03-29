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

def one_minus_FS_from_EC(targets, decisions, priors=None, beta=None):
    '''
    That is, 1 − Fβ is proportional to ECβ2 with a scaling factor given by the 
    inverse of β2P2 + R∗2.
    # OJO que tarda 1m45.3s para umbrales (-7,7,0.1)
    '''
    N1,N2,_,_,K12,K21 = utils.get_counts_from_binary_data(targets, decisions)
    P1 = priors[0]
    P2 = priors[1]
    costs_beta = ec.cost_matrix([[0, np.sqrt(beta)],[1,0]])
    R = utils.compute_R_matrix_from_counts_for_binary_classif(K12, K21, N1, N2)
    R_ast_2 = R[0][1]+R[1][1]
    
    NEC_beta = ec.average_cost_from_confusion_matrix(R, priors, costs_beta, adjusted=True)
    
    return min(np.sqrt(beta)*P2, P1)*(NEC_beta/(np.sqrt(beta)*P2+R_ast_2))


def mccoeff_from_EC(targets, decisions, costs=None , priors=None):
    '''
    '''
    _,_,K11,K22,K12,K21 = utils.get_counts_from_binary_data(targets, decisions)
    K1_ast = K11 + K12
    K2_ast = K21 + K22
    K_ast_1 = K11 + K21
    K_ast_2 = K12 + K22
    
    NEC_u = ec.average_cost(targets, decisions, costs, priors, adjusted=True)
    num = K1_ast*K2_ast
    den = K_ast_1*K_ast_2
    
    
        
    return (np.sqrt(num/den)*(1-NEC_u)) if den>0 else (np.inf if num>0 else -np.inf)


def lrplus_from_EC(targets, decisions, costs=None, priors=None):
    '''
    LR+ is the likelihood ratio for positive results. it is used for non-symmetric binary 
    classification problems where one of the classes is the class of interest. It is given by: 
    sensitivity / (1-specificity) if we assume class 2 is the class of interest, sensitivity = R22 
    and specificity = R11. 
    '''
    _,_,K11,K22,K12,K21 = utils.get_counts_from_binary_data(targets, decisions)
    R12 = utils.compute_R_matrix_from_counts_for_binary_classif(K12, K21, K11, K22)[0][1]
    NEC_u = ec.average_cost(targets, decisions, costs=costs, priors=priors, adjusted=True)

    return (((1-NEC_u)/R12)+1) if R12>0 else np.inf

#######################################################################
# Other ways of computing minDCF
    
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