# RBF-SCORE
## Description of this algorithm
This algorithm finds the optimal shaping parameter and condition number of RBF for community detection algorithm SCORE and its variants. We also optimize the SCORE+ algorithm with higher order proximity of the affinity matrix.

## How to run this code?
It is easy to to do this. 
1. Go to the `basic_function.py` under the folder "rbf-score". 
2. Scroll down and find the `main()` function. There is only one line of code that calls the `run_networks()`. You will be required to point out which network/dataset to run. For instance, you would like to run the football.mat data in "/data/datasets", just give a string "football" to the function.
3. Then run the code, done!

## Outputs
Your will get outputs like this if you run lesmis data
```python
run_networks(fn='lesmis')
```

```
c, condition number for optimal NMI - lesmis
2.449 1.2330039145526659e+20 :  0.751 0.306

c, condition number for optimal Modularity - lesmis
2.7551 1.8409452761073323e+20 :  0.708 0.423
```
As said above, the first two lines of the outputs tell us that if shaping parameter is 2.449, condition number will be 1.23e+20 with NMI=0.751 and Q=0.306. This is for the best NMI. Similarly, the best modularity 0.423 is acquired when shaping setting parameter as 2.7551.
