# data/


### patterns

Files matching the format `patterns_X-Y.npy` contain the dataset used to test the dQA dynamics with perceptron hamiltonian. This files have been generated via the `generator.py` script.
* `X` is the number of samples contained in the dataset (i.e. rows)
* `Y` is the number of features (i.e. columns)

Eventually, a number after the dot (for instance `patterns_17-21.1.npy`) will identify a different (random) realization of the same dataset.


### benchmarks

The directory `benchmark` will contain the data used to evaluate the performances of our models.


### external sources

Files `paper_*` will contain data taken from the reference paper.
