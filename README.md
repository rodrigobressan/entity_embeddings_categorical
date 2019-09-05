[![PyPI version](https://badge.fury.io/py/entity-embeddings-categorical.svg)](https://pypi.org/project/entity-embeddings-categorical)
[![Build Status](https://travis-ci.org/bresan/entity_embeddings_categorical.svg?branch=master)](https://travis-ci.org/bresan/entity_embeddings_categorical)
[![Coverage Status](https://coveralls.io/repos/github/bresan/entity_embeddings_categorical/badge.svg?branch=master)](https://coveralls.io/github/bresan/entity_embeddings_categorical?branch=master)
[![GitHub](https://img.shields.io/github/license/bresan/entity_embeddings_categorical.svg)](https://github.com/bresan/entity_embeddings_categorical/blob/master/LICENSE.md)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e02bc243822c4ce884c4adf87ff6e9f7)](https://www.codacy.com/app/bresan/entity_embeddings_categorical?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bresan/entity_embeddings_categorical&amp;utm_campaign=Badge_Grade)

# Overview

This project is aimed to serve as an utility tool for the preprocessing, training and extraction of entity embeddings through Neural Networks using the Keras framework. It's still under construction, so please use it carefully.

# Installation

The installation is pretty simple if you have a virtualenv already installed on your machine. If you don't please rely to [VirtualEnv official documentation](https://virtualenv.pypa.io/en/latest/).

```bash
pip install entity-embeddings-categorical
```

# Documentation

Besides the docstrings, major details about the documentation can be found [here](https://entity-embeddings-categorical.readthedocs.io/en/latest/).

# Testing

This project is inteded to suit most of the existent needs, so for this reason, testability is a major concern. Most of the code is heavily tested, along with [Travis](https://travis-ci.org/bresan/entity_embeddings_categorical) as Continuous Integration tool to run all the unit tests once there is a new commit.

# Usage

The usage of this utility library is provided in two modes: default and custom. In the default configuration, you can perform the following operations: Regression, Binary Classification and Multiclass Classification.

If your data type differs from any of these, you can feel free to use the custom mode, where you can define most of the configurations related to the target processing and output from the neural network.

## Default mode


The usage of the default mode is pretty straightforward, you just need to provide a few parameters to the Config object:

So for creating a simple embedding network that reads from file **sales_last_semester.csv**, where the target name is **total_sales**, with the desired output being a **binary classification** and with a training ratio of **0.9**, our Python script would look like this:

```python
    config = Config.make_default_config(csv_path='sales_last_semester.csv',
                                        target_name='total_sales',
                                        target_type=TargetType.BINARY_CLASSIFICATION,
                                        train_ratio=0.9)


    embedder = Embedder(config)
    embedder.perform_embedding()
```

Pretty simple, huh?

A working example of default mode can be found [here as a Python script](https://github.com/bresan/entity_embeddings_categorical/blob/master/example/default/default_config_example.py).

## Custom mode

If you intend to customize the output of the Neural Network or even the way that the target variables are processed, you need to specify these when creating the configuration object.
This can be done by creating a class that extend from [TargetProcessor](https://github.com/bresan/entity_embeddings_categorical/blob/master/entity_embeddings/processor/processor.py) and [ModelAssembler](https://github.com/bresan/entity_embeddings_categorical/blob/master/entity_embeddings/network/assembler.py).

A working example of custom configuration mode can be found [here](https://github.com/bresan/entity_embeddings_categorical/blob/master/example/custom/custom_config_example.py).

## Visualization

Once you are done with the training of your model, you can use the module [visualization_utils](https://github.com/bresan/entity_embeddings_categorical/blob/master/entity_embeddings/util/visualization_utils.py) in order to create some visualizations from the generated weights as well as the accuraccy of your model.

Below are some examples created for the [Rossmann dataset](https://www.kaggle.com/c/rossmann-store-sales):

![Weights for store id embedding](https://raw.githubusercontent.com/bresan/entity_embeddings_categorical/master/example/default/artifacts/visualizations/Store_embedding.png)

# Troubleshooting

In case of any issue with the project, or for further questions, do not hesitate to open an issue here on GitHub.

# Contributions

Contributions are really welcome, so feel free to open a pull request :-)

# TODO

- Allow to use a Pandas DataFrame instead of the csv file path;
