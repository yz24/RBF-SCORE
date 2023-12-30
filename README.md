# RBF-SCORE+
## Description
This algorithm finds the optimal shaping parameter and condition number of various graph RBF kernels for community detection algorithm SCORE+ and its variants. We also optimize the SCORE+ algorithm with higher order proximity of the affinity matrix.

## How to run? 
1. Go to the `basic_function.py` under the folder `rbf-score`. 
2. Scroll down and find the `main()` function. 
3. When running on real-world data sets, for instance, running the football.mat data under `/data/datasets` with a `MQ` RBF, just set `fn= 'football'` and `RBF='MQ'`.
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
The outputs tell us that the optimal shaping parameter is 0.303 and the corresponding condition number will be 289.3264, with NMI = 0.957 and Q = 0.62.

If you use this code, please cite:
```
@ARTICLE{10373106,
  author={Zhu, Yanhui and Hu, Fang and Kuo, Lei Hsin and Liu, Jia},
  journal={IEEE Transactions on Big Data}, 
  title={SCOREH+: A High-Order Node Proximity Spectral Clustering on Ratios-of-Eigenvectors Algorithm for Community Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TBDATA.2023.3346715}}

```
