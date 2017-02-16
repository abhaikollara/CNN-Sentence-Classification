# CNN-Sentence-Classification
A tensorflow implementation of Convolutional Neural Networks for Sentence Classification. The original paper can be found at https://arxiv.org/abs/1408.5882

The implementation differs from the original paper in the following ways : 
- Pretrained word vectors are not used
- Two seperate channels of word vectors are not used, all vectors are learned
- L2 norm constraints are not enforced

Edit the config file to change the configuration of the model

## Results
The model produces a test accuracy of 75.33 % within 2 epochs. The results produced in the paper for the given architecture is 76.1 %

## Comparison with other models
Comparison with other models (taken from the paper). The model could perform much better with pretrained word vectors

|_Architecture_|_Test accuracy_|
|:-------|:---:|
|**CNN-rand**|**76.1**|
|CNN-static| 81.0|
|CNN-non-static| 81.5|
|CNN-multichannel| 81.1|
|RAE (Socher et al., 2011)| 77.7|
|MV-RNN (Socher et al., 2012)| 79.0|
|CCAE (Hermann and Blunsom, 2013)| 77.8|
|Sent-Parser (Dong et al., 2014)| 79.5|
|NBSVM (Wang and Manning, 2012)| 79.4|
|MNB (Wang and Manning, 2012)| 79.0|
|G-Dropout (Wang and Manning, 2013)| 79.0|
|F-Dropout (Wang and Manning, 2013)| 79.1|
|Tree-CRF (Nakagawa et al., 2010)| 77.3|
