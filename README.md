
# Conditional Variational Autoencoder for Tree-Structured Data

The implementation developed along with my Master Thesis in Computer Science.

Citing the abstract:
>The thesis deals with the design of a deep learning model that can learn a generative process realizing unconstrained tree transductions. The model is based on an extension of the popular Variational Autoencoder framework to allow conditioning the generative process on tree-structured inputs and to generate tree-structured predictions. An efficient Tensor- Flow implementation of the proposed model has been realized and vali- dated on Arithmetic Expression trees and Neural Machine Translation.

For more details about the abstract model architecture and the main implementation choices see the full work [here](thesis.pdf) 

## Tensorflow Trees
Out of this work a library to deal with tree structured data have been refactored, see [m-colombo/tf_tree](https://github.com/m-colombo/tf_tree)

## Code Structure

- `tree/` contains the core support classes to handle trees
    - `batch`: classes supporting the tree storage for batched computations
    - `definition`: basic definitions needed to characterized the tree domain at hand
    - `simple_expression`: some tree characterizations of simple arithmetic expression tree domains
- `tree_encoder`, `tree_decoder`: core components implementing the tree computations
- `vae`: variational autoencoder implementation
- `benchmark\`, `data\`, `experiments\`: code used for the experiments, to run benchmark and preprocess data

## How To
See [`src/experiments/exp.py`](src/experiments/exp.py) for an example on how to run experiments.

Although this is the code used to run the experiments there might be some inconsistencies and undocumented facets. This is intended to be a reference implementation, possibly out of this an ad hoc library might be refactored some time in the future.

If you are willing to use this implementation and you need some clarifications, you'd like to have a clean library or you're just curious about something feel free to contact me opening an issue.

## Run Environments
The implementation have been tested with Tensorflow versions 1.11 and 1.12. In the variants GPU, CPU and CPU-MKL. 
As run environments we tested Ubuntu, Centos, MacOS X and Docker on all of them.
