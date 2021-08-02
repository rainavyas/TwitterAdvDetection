# TwitterAdvDetection
Compare detection approaches for semantic adversarial attacks on a Twitter sentiment classifier

# Objective

NLP classification of twitter tweets into one of six emotions: love, joy, fear, anger, surprise, sadness.
The dataset is described in https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt

The model is adversarially attacked using a saliency ranked synonym substitution attack approach. The aim is to detect the adversarial samples using a range of detection approaches suggested in literature


# Requirements

python3.4 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers


# Training

Fork the repository (and clone).

Run the _train.py_ scripts with desired arguments in your terminal. For example, to train an ELECTRA-based classifier:

_python ./train.py electra_trained.th electra --B=8 --lr=0.00001 --epochs=5 --suppression=0.1_

# Experimental Results

| Detection Approach | F1 Score |
| ----------------- | :-----------------: |
Ensemble Uncertainty |  |
Mahalanobis Distance | |

Detection was applied to a 6-word substitution attack on the Electra based Twitter emotion classifiers.
