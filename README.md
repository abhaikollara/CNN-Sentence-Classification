# CNN-Sentence-Classification
A tensorflow implementation of Convolutional Neural Networks for Sentence Classification. The original paper can be found at https://arxiv.org/abs/1408.5882

The implementation differs from the original paper in the following ways : 
- Pretrained word vectors are not used
- Two seperate channels of word vectors are not used, all vectors are learned
- L2 norm constraints are not enforced

Edit the config file to change the configuration of the model
