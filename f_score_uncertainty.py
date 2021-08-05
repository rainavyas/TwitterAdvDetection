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

Calculate the best F1-score for this metric to detect
adversarial samples.


Explanation of mutual information as uncertainty metric:

    We are considering I(Y;w|data), the mutual information
    between the output prediction (Y) and the model parameters (w)

    This is theoretically equal to entropy of Y (H(Y))
    minus expectaion of H(Y|w), where expectation is wrt p_w
    (i.e. the distribution of model parameters)

    An ensemble of models can be used to approximate the distribution
    for p_w... so we can approximate mutual information as:

        I(Y;w|data) = H(E(P(Y))) - E(H(P(Y)),
        where E refers to the expectation over ensemble of models,
        i.e. a simple average over 5 models in this case.
'''
import numpy as np

if __name__ == '__main__':
	# Get command line arguments
	commandLineParser = argparse.ArgumentParser()
	commandLineParser.add_argument('FILENAME', type=str, help='.npy file with logits array')

    args = commandLineParser.parse_args()
    filename = args.FILENAME

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/f_score_uncertainty.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the logits array
    logits = np.load(filename)
    print("Array shape", logits.shape)

    # Calculate E(P(Y))
