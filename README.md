# DisNet-2018-tfm

## Abstract:

Understanding the inner workings of deep learning algorithms is key to efficiently exploit the large number of videos that are generated every day. For the self-supervised learning of the spatio-temporal information contained within these videos, there are several types of algorithms based on convolutional neural networks (CNNs) following an auto-encoder style architecture. However, we have checked that this type of models, trained for the frame prediction task, learn jointly these spatio-temporal information, so the model is not able to recognize appearance-motion combinations not seen during training. Our proposed model, called DisNet, can learn separately the appearance and motion through disentanglement, so that it solves the generalization and scalability problems. To demonstrate this, we conducted numerous experiments under highly controlled conditions, generating specific datasets that make the "conventional" model fails for the appearance and motion classification tasks, and analyzing how well our proposal behaves under the same conditions.

**Keywords:** deep learning, convolutional neural networks, auto-encoders, disentanglement, motion, appearance.
