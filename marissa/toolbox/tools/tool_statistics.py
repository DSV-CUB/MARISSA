from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats
import numpy as np


def confusion_matrix_info(P, N, PP, PN, TP, TN, FP, FN):
    result = {}
    result["P"] = P
    result["N"] = N
    result["PP"] = PP
    result["PN"] = PN
    result["TP"] = TP
    result["TN"] = TN
    result["FP"] = FP
    result["FN"] = FN
    result["TPR"] = 100 * TP / P # true positive rate / recall / sensitivity
    result["FNR"] = 100 * FN / P # false negative rate
    result["FPR"] = 100 * FP / N # false positive rate
    result["TNR"] = 100 * TN / N # true negative rate / specifity / selectivity
    result["BM"] = result["TPR"] + result["TNR"] - 100 # bookmaker informedness
    result["PT"] = 100 * (np.sqrt(result["TPR"] * result["FPR"]) - result["FPR"]) / (result["TPR"] - result["FPR"]) # prevalence threshold
    result["Prevalence"] = (100 * P) / (P + N) # prevalence
    result["ACC"] = (100 * (TP + TN)) / (P + N) # accuracy
    result["BA"] = (result["TPR"] + result["TNR"]) / 2 # balanced accuracy
    result["PPV"] = 100 * TP / PP # positive predictive value / precision
    result["FOR"] = 100 * FN / PN # false omission rate
    result["FDR"] = 100 * FP / PP # false discovery rate
    result["NPV"] = 100 * TN / PN # negative predictive value
    result["F1score"] = 100* 2 * TP / (2* TP + FP + FN) # F1 score
    result["FM"] = np.sqrt(result["PPV"] * result["TPR"]) # Fowlkes Mallows Index
    result["LR+"] = result["TPR"] / result["FPR"] # positive liklihood ratio
    result["LR-"] = result["FNR"] / result["TNR"] # negative likelihood ratio
    result["MK"] = result["PPV"] + result["NPV"] - 1 # Markedness / deltaP
    result["DOR"] = result["LR+"] / result["LR-"] # diagnostic odds ratio
    result["MCC"] = np.sqrt(result["TPR"] * result["TNR"] * result["PPV"] * result["NPV"]) - np.sqrt(result["FNR"] * result["FPR"] * result["FOR"] * result["FDR"]) # Matthews correlation coefficient
    result["TS"] = 100 * TP / (TP + FN + FP) # Threat Score / Critical Success Score / Jaccard Index
    return result

def roc_curve_analysis(x_positive, x_negative, optimum="np.argmax(tpr-fpr)"):
    xp = np.array(x_positive).flatten()
    xn = np.array(x_negative).flatten()

    P = 100 * (len(xp) / (len(xp) + len(xn)))
    N = 100 * (len(xn) / (len(xp) + len(xn)))

    if np.mean(xp) < np.mean(xn):
        xn_copy = np.copy(xn)
        xn = np.copy(xp)
        xp = xn_copy
        do_swap = True
    else:
        do_swap = False

    y_scores = np.concatenate((xp, xn), axis=0)
    y_labels = np.concatenate((np.ones(len(xp)), np.zeros(len(xn))), axis=0)

    fpr, tpr, threshholds = roc_curve(y_labels, y_scores, pos_label=1)
    index_optimum = eval(optimum)
    try:
        index_optimum = index_optimum.flatten()[0]
    except:
        pass


    if do_swap:
        prediction = y_scores<=threshholds[index_optimum]
        reality = np.logical_not(y_labels.astype(bool))
        #FN = 100 * (np.sum(np.logical_and(reality, prediction)) / len(reality))
        #FP = 100 * (np.sum(np.logical_and(np.logical_not(reality), np.logical_not(prediction))) / len(reality))
        #TN = 100 * (np.sum(np.logical_and(np.logical_not(reality), prediction)) / len(reality))
        #TP = 100 * (np.sum(np.logical_and(reality, np.logical_not(prediction))) / len(reality))
    else:
        prediction = y_scores>=threshholds[index_optimum]
        reality = y_labels.astype(bool)

    TP = 100 * (np.sum(np.logical_and(reality, prediction)) / len(reality))
    TN = 100 * (np.sum(np.logical_and(np.logical_not(reality), np.logical_not(prediction))) / len(reality))
    FP = 100 * (np.sum(np.logical_and(np.logical_not(reality), prediction)) / len(reality))
    FN = 100 * (np.sum(np.logical_and(reality, np.logical_not(prediction))) / len(reality))
    PP = 100 * (np.sum(prediction) / len(reality))
    PN = 100 - PP


#    from sklearn.metrics import RocCurveDisplay
#    from matplotlib import pyplot as plt
#    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
#    plt.show()

    result = {}
    result["roc_tpr"] = tpr
    result["roc_fpr"] = fpr
    result["roc_threshholds"] = threshholds
    result["roc_auc"] = roc_auc_score(y_labels, y_scores)
    result["optimal_threshhold"] = threshholds[index_optimum]
    result["optimal_roc_point"] = (fpr[index_optimum], tpr[index_optimum])
    result["optimal_confusion_matrix"] = confusion_matrix_info(P, N, PP, PN, TP, TN, FP, FN)
    return result


def get_confidence_interval(data, alpha=0.05):
    u = np.mean(data)
    s = np.std(data)
    l = len(data)

    #if len(data) > 30:
    #    z = stats.t.ppf((1-(alpha/2)), df=np.inf)
    #else:
    z = stats.t.ppf((1-(alpha/2)), df=l-1)

    CI_low = u - (z * s / np.sqrt(l))
    CI_high = u + (z * s / np.sqrt(l))

    return CI_low, CI_high