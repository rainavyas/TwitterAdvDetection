'''
Expected a tensor of logits at the input: [N x 5 x 2 x 6]

    Dim 0: Data point
            [N] # Number of successful adversarial attacks

    Dim 1: Seed
            [5] # 5 models in ensemble

    Dim 2: Original or Adversarial
            [2]

    Dim 3: Emotion Logit predictions
            [6] # 6 emotions in classification task

Use an uncertainty metric (mutual information in this case)
to caluclate uncertainty for each original and adv sample.
Note that the mutual information is meant to indicate the model,
epistemic uncertainty.

Calculate the best F1-score for this metric to detect
adversarial samples.


Explanation of mutual information as uncertainty metric:

    We are considering I(Y;w|data), the mutual information
    between the output prediction (Y) and the model parameters (w)

    This is theoretically equal to entropy of Y (H(Y))
    minus expectaion of H(Y|w), where expectation is wrt p_w
    (i.e. the distribution of model parameters)
    This is called: predictive entropy - expected entropy

    An ensemble of models can be used to approximate the distribution
    for p_w... so we can approximate mutual information as:

        I(Y;w|data) = H(E(P(Y))) - E(H(P(Y)),
        where E refers to the expectation over ensemble of models,
        i.e. a simple average over 5 models in this case.
'''

import sys
import os
import argparse
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def get_best_f_score(precisions, recalls, beta=1.0):
    f_scores = (1+beta**2)*((precisions*recalls)/((precision*(beta**2))+recall))
    ind = np.argmax(f_scores)
    return precisions[ind], recalls[ind], f_scores[ind]

if __name__ == '__main__':
	# Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('FILENAME', type=str, help='.npy file with logits array')
    commandLineParser.add_argument('OUT', type=str, help='.png file for pr curve')

    args = commandLineParser.parse_args()
    filename = args.FILENAME
    out_file = args.OUT

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/f_score_uncertainty.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the logits array
    logits = np.load(filename)
    probs = softmax(logits, axis=-1)
    print("Array shape", probs.shape)

    original_probs = probs[:,:,0,:].squeeze()
    adv_probs = probs[:,:,1,:].squeeze()

    # Calculate E(P(Y))
    original_epy = np.mean(original_probs, axis=1)
    adv_epy = np.mean(adv_probs, axis=1)

    # Determine H(E(P(Y)))
    original_hepy = entropy(original_epy, axis=-1)
    adv_hepy = entropy(adv_epy, axis=-1)

    # Calculate E(H(P(Y)))
    original_ehpy = np.mean(entropy(original_probs, axis=-1), axis=1)
    adv_ehpy = np.mean(entropy(adv_probs, axis=-1), axis=1)

    # Determine the mutual informations
    original_I = original_hepy - original_ehpy
    adv_I = adv_hepy - adv_ehpy

    Is = np.concatenate((original_I, adv_I))
    labels = [0]*len(original_I) + [1]*len(adv_I)

    # Calculate best F1-score
    precision, recall, _ = precision_recall_curve(labels, Is)
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    # plot all the data
    plt.plot(recall, precision, 'r-')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(out_file)
