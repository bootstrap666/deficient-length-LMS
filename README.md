# A Linearly Constrained Framework for the Analysis of the Deficient Length Least-Mean Square Algorithm

This is the github repo for the paper ["A Linearly Constrained Framework for the Analysis of the Deficient Length Least-Mean Square Algorithm"](https://doi.org/10.1016/j.dsp.2024.104747) by M.H. Maruo and J.C.M. Bermudez accepted for publication at ["Digital Signal Processing (ISSN 1051-2004)"](https://www.sciencedirect.com/journal/digital-signal-processing). You may want to quickstart with [deficient%20length%20LMS%20test.ipynb](https://github.com/bootstrap666/deficient-length-LMS/blob/main/deficient%20length%20LMS%20test.ipynb)

Most adaptive system identification analyses  assume the length of the adaptive filter to match the length of the unknown system response. This assumption tends to be unrealistic, and its conclusions may not apply to some practical applications. The behavior of deficient length adaptive filters has been studied using different approaches. This work formulates the deficient length adaptive system problem using a  Linearly Constrained Minimum Mean Squared Error (LCMMSE) framework. This new formulation leads to a very interesting interpretation that allows the utilization of results from the study of constrained adaptive filters to understand the behavior of the deficient length LMS adaptive filter.  The reduced number of coefficients is formulated as a linear optimization constraint that defines a projection onto the feasible space for the adaptive weight vector. We derive analytical models for the mean and mean-square behavior of the adaptive weights. In addition, we derive an analytical model for the variance of the steady-state squared estimation error which provides important design information.  Simulation results show excellent matching between theory and actual algorithm behavior. 

## Computation requirements

Monte-Carlo simulations code was written using [Numba](https://numba.pydata.org/) for parallelization.

## Citation

```python
@article{MARUO2024104747,
title = {A Linearly Constrained Framework for the Analysis of the Deficient Length Least-Mean Square Algorithm},
journal = {Digital Signal Processing},
pages = {104747},
year = {2024},
issn = {1051-2004},
doi = {https://doi.org/10.1016/j.dsp.2024.104747},
url = {https://www.sciencedirect.com/science/article/pii/S1051200424003725},
author = {Marcos H. Maruo and José C.M. Bermudez},
keywords = {Adaptive filtering, least mean-square (LMS) algorithm, deficient length, statistical analysis},
abstract = {Most adaptive system identification analyses assume the length of the adaptive filter to match the length of the unknown system response. This assumption tends to be unrealistic, and its conclusions may not apply to some practical applications. The behavior of deficient length adaptive filters has been studied using different approaches. This work formulates the deficient length adaptive system problem using a Linearly Constrained Minimum Mean Squared Error (LCMMSE) framework. This new formulation leads to a very interesting interpretation that allows the utilization of results from the study of constrained adaptive filters to understand the behavior of the deficient length LMS adaptive filter. The reduced number of coefficients is formulated as a linear optimization constraint that defines a projection onto the feasible space for the adaptive weight vector. We derive analytical models for the mean and mean-square behavior of the adaptive weights. In addition, we derive an analytical model for the variance of the steady-state squared estimation error which provides important design information. Simulation results show excellent matching between theory and actual algorithm behavior.}
}
```
