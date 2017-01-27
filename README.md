# CNN-Sentence-Classification
A tensorflow implementation of Convolutional Neural Networks for Sentence Classification. The original paper can be found at https://arxiv.org/abs/1408.5882

The code follows an article published at http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/. Therefore the implementation differs from the original paper in the following ways : 
- Pretrained word vectors are not used
- Two seperate channels of word vectors are not used, all vectors are learned
- L2 norm constraints are not enforced
