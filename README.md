# Ensembling parameters with differential evolution

This repository shows how to ensemble parameters of two trained neural networks using differential evolution. The steps
followed are as follows:

* Train two networks (architecturally same) on the same dataset (CIFAR-10 used here) but from two different random 
initializations.
* Ensemble their weights using the following formulae:
    
    ```py
    w_t = w_o * ema + (1 - ema) * w_p
    ```
  `w_o` and `w_p` represents the learned of a neural network.
* Randomly initialize a network (same architecture as above) and populate its parameters `w_t` using the above
formulae.

`ema` is usually chosen by the developer in an empirical manner. This project uses differential evolution to find it.

Below are the top-1 accuracies (on CIFAR-10 test set) of two individually trained two models along with their
ensembled variant:

* Model one: 63.23%
* Model two: 63.42%
* Ensembled: 63.35%

With the more conventional average prediction ensembling, I was able to get to 64.92%. This is way better than what I got
by ensembling the parameters. Nevertheless, the purpose of this project was to just try out an idea. 

## Reproducing the results

Ensure the `requirements.txt` is satisfied. Then train two models with ensuring your working directory is at the root
of this project:

```shell
$ git clone https://github.com/sayakpaul/parameter-ensemble-differential-evolution
$ cd parameter-ensemble-differential-evolution
$ pip install -qr requirements.txt
$ for i in `seq 1 2`; python train.py; done
```

Then just follow the `ensemble-parameters.ipynb` notebook. You can also use the networks I trained. Instructions are
available inside the notebook. 

## References

* https://en.wikipedia.org/wiki/Differential_evolution
* https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
* https://machinelearningmastery.com/model-averaging-ensemble-for-deep-learning-neural-networks/



