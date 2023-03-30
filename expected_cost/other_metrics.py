import numpy as np
from sklearn.metrics import roc_curve, det_curve, recall_score
from expected_cost import ec, utils


#########################################################################################
# Definition of a few standard metrics computed from the confusion matrix

def f_score(K10, K01, N0, N1):
    K11 = N1-K10
    Recall    = K11/N1
    Precision = K11/(K11+K01) if K11+K01>0 else 0
    # Returns Fscore
    return 2 * Precision*Recall/(Recall+Precision) if K11>0 else 0

def mc_coeff(K10, K01, N0, N1):
    K11 = N1-K10
    K00 = N0-K01
    num = K00 * K11 - K01 * K10
    den = np.sqrt(N0 * N1 * (K01+K11) * (K10 + K00))
    return num/den if den>0 else (np.inf if num>0 else -np.inf)

def lr_plus(K10, K01, N0, N1):
    R10 = K10 / N1
    R01 = K01 / N0
    return (1-R10)/R01 if R01>0 else np.inf

def accuracy(K00, K11, K):
    return (K00+K11)/K

def bal_accuracy(C, K00, K11, N0, N1):
    return 1/C*(K00/N0+K11/N1)

def net_benefit(K01, K11, K, pt=None):
    factor = pt/(1-pt)
    return (K11/K)-(factor*(K01/K))


#########################################################################################
# Other metrics computed using EC so that is obvious that they are specific cases of EC. 

def accuracy_from_EC(targets, decisions):
    # Number of classes 
    C = len(np.unique(targets))
    # Prior for class 0
    p0 = utils.get_binary_data_priors(targets)[0]
    # Prior vector 
    data_priors = np.array([p0] + [(1 - p0) / (C - 1)] * (C - 1))
    # Cost matrix
    costs_01 = ec.cost_matrix.zero_one_costs(C)
    # Expected cost
    EC = ec.average_cost(targets, decisions, costs_01, data_priors, adjusted=False)
    return 1-EC

def bal_accuracy_from_EC(targets, decisions):
    # Number of classes 
    C = len(np.unique(targets))
    # Prior for class 0
    p0 = utils.get_binary_data_priors(targets)[0]
    # Prior vector 
    data_priors = np.array([p0] + [(1 - p0) / (C - 1)] * (C - 1))
    # A cost matrix for balance accuracy with c_ij = 1/(CP_i)
    bal_acc_cost = ec.cost_matrix([[0, 1/(C*data_priors[0])], [1/(C*data_priors[1]), 0]])
    EC = ec.average_cost(targets, decisions, bal_acc_cost, data_priors, adjusted=False)
    return 1-EC

def net_benefit_from_EC(targets, decisions, pt=None):
    # Number of classes 
    C = len(np.unique(targets))
    # Prior for class 0
    p0 = utils.get_binary_data_priors(targets)[0]
    # Prior vector 
    data_priors = np.array([p0] + [(1 - p0) / (C - 1)] * (C - 1))
    c12 = pt/(1-pt)
    # Cost matrix
    costs_nb = ec.cost_matrix([[0, c12],[1,0]])
    net_factor = data_priors[1] - min(data_priors[1], (pt/1-pt)*data_priors[0])
    NEC = ec.average_cost(targets, decisions, costs_nb, data_priors, adjusted=True)
    return net_factor * NEC

def one_minus_fscore_from_EC(targets, decisions, beta=None):
    # Number of classes 
    C = len(np.unique(targets))
    # Prior for class 0 where
    P1 = utils.get_binary_data_priors(targets)[1]
    # All data priors 
    priors = np.array([P1] + [(1 - P1) / (C - 1)] * (C - 1))
    P2 = priors[0]
    # Counts and R matrix   
    N1,N2,_,_,K12,K21 = utils.get_counts_from_binary_data(targets, decisions)
    R = utils.compute_R_matrix_from_counts_for_binary_classif(K12, K21, N1, N2)
    R_ast_2 = R[0][1]+R[1][1]
    # Cost matrix
    costs_beta = ec.cost_matrix([[0, beta**2],[1,0]])
    EC_beta = ec.average_cost_from_confusion_matrix(R, priors, costs_beta, adjusted=False)
    #NEC_beta = ec.average_cost_from_confusion_matrix(R, priors, costs_beta, adjusted=True)
    return EC_beta / ((beta**2*P2)+R_ast_2)
    #return min(beta**2*P2, P1)*(NEC_beta/((beta**2*P2)+R_ast_2))

def mccoeff_from_EC(targets, decisions):
    # Number of classes 
    C = len(np.unique(targets))
    # Uniform priors
    unif_priors = np.ones(C) / C
    # Counts
    _,_,K11,K22,K12,K21 = utils.get_counts_from_binary_data(targets, decisions)
    K1_ast = K11 + K12
    K2_ast = K21 + K22
    K_ast_1 = K11 + K21
    K_ast_2 = K12 + K22
    # We consider the usual 0-1 cost matrix
    costs_01 = ec.cost_matrix.zero_one_costs(C)
    NEC_u = ec.average_cost(targets, decisions, costs_01, unif_priors, adjusted=True)
    # Factor
    num = K1_ast*K2_ast
    den = K_ast_1*K_ast_2
    return (np.sqrt(num/den)*(1-NEC_u)) if den>0 else (np.inf if num>0 else -np.inf)

def lrplus_from_EC(targets, decisions):
    # Number of classes 
    C = len(np.unique(targets))
    # Uniform priors
    unif_priors = np.ones(C) / C
    # Counts 
    N1,_,_,_,K12,_ = utils.get_counts_from_binary_data(targets, decisions)
    R12 = K12 / N1
    # Cost matrix 
    costs_01 = ec.cost_matrix.zero_one_costs(C)
    NEC_u = ec.average_cost(targets, decisions, costs_01, unif_priors, adjusted=True)
    return (((1-NEC_u)/R12)+1) if R12>0 else np.inf


##########################################################################################
# Other ways of computing minDCF

# MinDCF from PyBOSARIS
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

# Sklearn 
def compute_det_curve(labels, scores, figure=0, pathfigure='det.png'):    
    fpr, fnr, thresholds = det_curve(np.array(labels,dtype=float), np.array(scores,dtype=float))
    return fpr, fnr, thresholds

# NetBenefit by Dayana Ribas using sklearn 
def nb(targets, scores, th=0.5):
    p1 = len(targets[targets == 1])/len(targets)
    decisions = (scores >= th).astype(int)
    sens = recall_score(targets, decisions)
    spec = recall_score(targets, decisions, pos_label=1)
    return (sens*p1 - (1-spec)*(1-p1)*th/(1-th))/p1
    