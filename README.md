# moon-and-stars-watermelon

WRITEUP:  
https://www.overleaf.com/1424963973fbdcxrqsjkhd

our paper: 
exmaple paper https://www.aclweb.org/anthology/P12-2018  
Classification of Fashion Article Images usingConvolutional Neural Networks [in PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8313740)  
https://ieeexplore.ieee.org/abstract/document/8313740



zalandoresearch/fashion-mnist  
https://github.com/zalandoresearch/fashion-mnist


Benchmark dahboard  
http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/

our result from SVC:  

|  C-value\Kernel | 'rbf'    | 'poly'  |
| -------------   |:-------------:| -----:|
|   0.1         | 0.8475|  |
|   1.0         | 0.8836 | 0.8755 |
|   10.0      | 0.8986 (89.70 from paper)     |   0.8934 |
| 100.0     | 0.8941      |    0.8913|
| 1000.0     |  0.8941    |   |

kernel='poly'  

|  degree |  1 |  2 | 3  | 4  |   5|	6|	7|	8	|9|
|---|---|---|---|---|---|---|---|---|---|
|C=100.0|0.8554|0.891	|0.8913|	0.8893|	0.8797|	0.8597	|0.8385|--|---|
|C=10.0|0.8449|	0.895	|0.8934	|0.8834	|0.8655	|0.8272	|0.7922		|0.7513|0.7219|  


**to do:  
table: compare our results with results from orginal paper if we are doing CNN**
