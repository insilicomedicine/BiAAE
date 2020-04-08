# Training scripts

# Noisy MNIST scripts

## Pretraining classifier

Before running conditional generation experiment on Noist MNIST, 
it's necessary to pretrain classifier, that will be used to compute validation metrics.

To pretrain classifier, please run

```
$ python pretrain_mnist_clf.py
```

#### Parameters:
`--gpu=` - gpu for training, if `-1` then will train on CPU

### Training models

To train some generative model on Noisy MNIST dataset, please, run following command.

```
$ python mnist_condgen_experiment.py
```

#### Parameters:
`--model=` - can be `biaae`, `uniaae`, `lat_saae`, `saae`, `cvae`, `jmvae`, `vib` or `vcca`

`--gpu=` - gpu for training, if `-1` then will train on CPU

-------

# LINCS scripts

## Pretraining

Before running training of model, please, pretrain RNN encoder and decoder.

To pretrain them, please run

```
$ python pretrain_rnn_enc_dec.py
```

#### Parameters:
`--gpu=` - gpu for training, if `-1` then will train on CPU

### Training models

To train conditional generative model (generate molecule by given transcriptome change) on LINCS dataset, 
please, run following command.

```
$ python lincs_experiment.py
```

#### Parameters:
`--model=` - can be `biaae`, `uniaae`, `lat_saae`, `saae`, `cvae`, `jmvae`, `vib` or `vcca`

`--gpu=` - gpu for training, if `-1` then will train on CPU

To train conditional generative model (generate transcriptome change by given molecule) on LINCS dataset, 
please, run following command.

```
$ python lincs_experiment_reverse.py
```

#### Parameters:
`--model=` - can be `biaae`, `uniaae`, `lat_saae`, `saae`, `cvae`, `jmvae`, `vib` or `vcca`

`--gpu=` - gpu for training, if `-1` then will train on CPU

