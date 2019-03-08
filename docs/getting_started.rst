Getting Started
==================


Installation
------------------
The installation is pretty simple if you have a virtualenv already installed on your machine. If you don't please rely to VirtualEnv official documentation.

.. code-block:: bash
    pip install entity-embeddings-categorical


Usage
=============

The usage of this utility library is provided in two modes: default and custom. In the default configuration, you can perform the following operations: Regression, Binary Classification and Multiclass Classification.

If your data type differs from any of these, you can feel free to use the custom mode, where you can define most of the configurations related to the target processing and output from the neural network.

Default
------------

The usage of the default mode is pretty straightforward, you just need to provide a few parameters to the Config object:

So for creating a simple embedding network that reads from file sales_last_semester.csv, where the target name is total_sales, with the desired output being a binary classification and with a training ratio of 0.9, our Python script would look like this:

.. code-block:: python
    config = Config.make_default_config(csv_path='sales_last_semester.csv',
                                        target_name='total_sales',
                                        target_type=TargetType.BINARY_CLASSIFICATION,
                                        train_ratio=0.9)


    embedder = Embedder(config)
    embedder.perform_embedding()


Custom
------------

If you intend to customize the output of the Neural Network or even the way that the target variables are processed, you need to specify these when creating the configuration object. This can be done by creating a class that extend from TargetProcessor and ModelAssembler.


Visualization
--------------

Once you are done with the training of your model, you can use the module visualization_utils in order to create some visualizations from the generated weights.

Troubleshooting
----------------

In case of any issue with the project, or for further questions, do not hesitate to open an issue on GitHub.

