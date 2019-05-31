Abstract
This project attempts to reproduce as well as improve the baseline accuracy from Bhatnagar,
Ghosal and Kolekar’s paper for the classification task of fashion article images obtained from
the Fashion-MNIST dataset.  In the first part of the project, we attempt to reproduce the SVC
baseline for the classification task, and obtained an accuracy score of
0.8986
, which is slightly
higher than the accuracy of
0.8970
in the original paper.  Then, we tried to improve the SVC
baseline  by  hyper-parameter  tuning,  involving  different  SVC  kernel  trials,  different  C-values
and different gamma values of the SVC models.  Our best performing model with polynomial
kernel (degree = 2 and
C
= 10
.
0) achieved a test accuracy of
0.8950
whereas the best performing
classifier with the RBF kernel (gamma = 0.001 and
C
= 10
.
0) resulted in an accuracy of
0.8970
.
We also implemented a Convolutional Neural Network model that has around 1% increase in
accuracy  compared  to  the  CNN  models  in  the  original  paper,  with  a  test  accuracy  score  of
0.9355
