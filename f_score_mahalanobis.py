'''
Expected a tensor of logits at the input: C x [N x 5 x 2 x 6]

    List Dim: Class
            [C] # 6 - label of emotion classes in order: [love, joy, fear, anger, surprise, sadness]

    Dim 0: Data point
            [N] # Number of successful adversarial attacks for class

    Dim 1: Seed
            [5] # 5 models in ensemble

    Dim 2: Original or Adversarial
            [2]

    Dim 3: Emotion Logit predictions
            [6] # 6 emotions in classification task


Use the Mahalanobis distance in the logit space as a metric for
each original and adv sample.

Calculate the best F1-score for this metric to detect
adversarial samples.
'''

import numpy as np
import sys
import os
import argparse

if __name__ == '__main__':
	# Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('FILENAME', type=str, help='.npz file with list of logits array')
    commandLineParser.add_argument('OUT', type=str, help='.png file for pr curve')
    commandLineParser.add_argument('--num_test', type=str, default='no', help="number of data points to use to test detector")

    args = commandLineParser.parse_args()
    filename = args.FILENAME
    out_file = args.OUT
    num_test = args.num_test

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/f_score_mahalanobis.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the logits array list
    # Keep only seed 1 logits (only model used for detector in logit space)
    logits_dict = np.load(filename)
    train_original_logits_list = []
    test_original_logits_list = []
    test_adv_logits_list = []
    TRAIN_FRAC = 0.8
    for i in range(6):
        logits = np.squeeze(logits_dict[f'arr_{i}'][:,0,:,:])
