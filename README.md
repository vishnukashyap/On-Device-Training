# On-Device-Training
Attempt at implementation of the paper Enabling On-Device CNN Training by Self-Supervised Instance Filtering and Error Map Pruning
https://arxiv.org/pdf/2007.03213.pdf

config.yaml contains all the values of the training parameters
Both standard training and training using the method described in the paper are present in train.py.

To run start the training, the command is:
  python train.py --training_type training type value

The training type can take value as standard or eif_emp
