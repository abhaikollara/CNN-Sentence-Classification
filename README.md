# CNN-Sentence-Classification
A tensorflow implementation of Convolutional Neural Networks for Sentence Classification. The original paper can be found at https://arxiv.org/abs/1408.5882

The implementation differs from the original paper in the following ways : 
- Pretrained word vectors are not used
- Two seperate channels of word vectors are not used, all vectors are learned
- L2 norm constraints are not enforced

Edit the config file to change the configuration of the model

## Results
The model produces a test accuracy of 75.33 % within 2 epochs. The results produced in the paper for the given architecture is 76.1 %
