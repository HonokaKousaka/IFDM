# Individually Fair Diversity Maximization
This is the official repository of the **NeurIPS 2025** paper submission titled *Individually Fair Diversity Maximization*.

## Setup

This implementation is based on Python 3. To run the code, you need the following dependencies.

- numpy==2.1.3
- gurobipy==12.0.2

## Repository structure
We select some important files for detailed description.

```python
|-- dataset # adjacency matrices of 4 datasets from graphic perspective
    |-- CelebA_complete.npy # the adjacency matrix of CelebA dataset
    |-- ...
|-- original_dataset # 4 datasets, each with 1,000 elements
    |-- CelebA.npy # CelebA dataset with 1,000 elements
    |-- ...
|-- max_min_opt.py # obtain the optimal max-min diversity through Gurobi
|-- max_sum_opt.py # obtain the optimal max-sum diversity through Gurobi
|-- sum_min_opt.py # obtain the optimal sum-min diversity through Gurobi
|-- max_min.py # obtain the approximate max-min diversity through our algorithm
|-- max_sum_greedy.py # obtain the approximate max-sum diversity through our algorithm
|-- sum_min.py # obtain the approximate sum-min diversity through our algorithm
|-- individual_max_min.py # obtain the max-min results indicating the effectiveness of fairness
|-- individual_max_sum.py # obtain the max-sum results indicating the effectiveness of fairness
|-- individual_sum_min.py # obtain the sum-min results indicating the effectiveness of fairness
```

Specifically, the results generated from `max_min_opt.py`, `max_sum_opt.py`, `sum_min_opt.py`, `max_min.py`, `max_sum_greedy.py`, `sum_min.py` are demomstrated in Section Experiment in our paper, while the results generated from `individual_max_min.py`, `individual_max_sum.py`, `individual_sum_min.py` are shown in Appendix in our paper.

## Run our code

If you want to reproduce the results in the paper, you can run our code in the following steps.

1. Make sure there are 2 directories named ```results``` and ```results_extended``` in the same directory that our code are in.

2. You can run our code as in the script in the below: 
```python
python max_min.py
python max_min_opt.py
python max_sum_greedy.py
python max_sum_opt.py
python sum_min.py
python sum_min_opt.py
python individual_max_min.py
python individual_max_sum.py
python individual_sum_min.py
```
