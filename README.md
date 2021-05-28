# RBF-SCORE+
## Description of this algorithm
This algorithm finds the optimal shaping parameter and condition number of various graph RBF kernels for community detection algorithm SCORE+ and its variants. We also optimize the SCORE+ algorithm with higher order proximity of the affinity matrix.

## How to run this code? 
1. Go to the `basic_function.py` under the folder "rbf-score". 
2. Scroll down and find the `main()` function. 
3. When running on real-world data sets, for instance, running the football.mat data with a `MQ` RBF in `/data/datasets`, just set fn= "football" and RBF='MQ'.
```python
 '''
Run real-world data set
fn: data name
RBF: RBFs, {'MQ', 'iMQ', 'gaussian'}
'''
data, y, k = import_real_data(fn='football')
run_networks(data, y, k, RBF='MQ')
 ```
4. Then run the code, done!

## Outputs
Your will get outputs like this if you run `football` data.
```python
0.303 289.3264 0.957 0.62
 ```
The outputs tell us that the optimal shaping parameter is 0.303 and the correspomdomh condition number will be 289.3264, with NMI=0.957 and Q=0.62.
