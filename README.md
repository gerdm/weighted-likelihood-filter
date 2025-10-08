# Weighted likelihood filter

![intro-plot](https://github.com/gerdm/weighted-likelihood-filter/assets/4108759/f32c0b01-433b-46e6-9153-262e1b6c4f10)

## Citation

```bib
@inproceedings{duran2024outlier,
  title={Outlier-robust Kalman filtering through generalised Bayes},
  author={Duran-Martin, Gerardo and Altamirano, Matias and Shestopaloff, Alexander Y and S{\'a}nchez-Betancourt, Leandro and Knoblauch, Jeremias and Jones, Matt and Briol, Fran{\c{c}}ois-Xavier and Murphy, Kevin},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={12138--12171},
  year={2024}
}
```

## Installation
To run the experiments, make sure to have installed `jax>=0.4.2`,
[rebayes-mini](https://github.com/gerdm/rebayes-mini/tree/main),
[flax](https://github.com/google/flax),
and the [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization) package:

```bash
pip install git+https://github.com/gerdm/rebayes-mini.git
pip install flax
pip install bayesian-optimization
```
