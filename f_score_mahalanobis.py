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

Use the Mahalanobis distance in the logit space as a metric for
each original and adv sample.

Calculate the best F1-score for this metric to detect
adversarial samples.
'''
