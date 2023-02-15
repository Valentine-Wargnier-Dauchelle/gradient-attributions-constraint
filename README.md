# A Weakly-Supervised Gradient Attribution Constraint for Interpretable Classification and Anomaly Detection

*Please cite : A Weakly-Supervised Gradient Attribution Constraint for Interpretable Classification and Anomaly Detection 
V. Wargnier-Dauchelle, T.Grenier, F. Durand-Dubief, F.Cotton, M. Sdika
Paper submitted to IEEE TMI*
 
**Abstract**: The lack of interpretability of deep learning hinders its practical use in clinics as critical applications require methods that are both accurate and transparent.
It also makes difficult to understand what is happening when a network does not work as expected, overfits, behaves strangely, etc.
For example, a healthy vs pathological classification model should rely on clinical features such as lesions or tumors and not on some training dataset biases.
A large number of post-hoc explanation models have been proposed to explain the decision of a trained network.
However they are very seldom used to enforce interpretability at training time and none in accordance with the classification.  
In this paper, we propose a new weakly-supervised method for both interpretable classification of healthy vs pathological subject and anomaly detection.
A new loss function is added to a standard classification model to constrain each voxel of healthy images to drive the network decision toward the healthy class according to some gradient-based attributions.
We show that this reveals pathological structures for patient images, allowing their unsupervised segmentation.
Moreover, we advocate both theoretically and experimentally that constrained training with the simple Gradient attribution map is similar to constraints with the heavier Expected Gradient\green{, consequently reducing the computationnal cost}.
We also propose a combination of attribution maps during the constrained training that makes the model robust to the choice of attribution maps at inference.
Our proposition was evaluated on two brain pathologies: tumors and multiple sclerosis.
In our experiments, this new constraint results in a more relevant classification, with a more pathology-driven decision.
For anomaly detection, the proposed method outperforms state-of-the-art especially on difficult multiple sclerosis lesions segmentation task with a 15 points Dice improvement. 

