# readream-motor

Codes used to write a Master's Thesis titled "A Neural Network to Detect Motor Imagery in ECoG Data Recorded During Dreaming".

The purpose of this Thesis was to develop a CNN-based predictor to discriminate between movement imagery types in a continuous brain activity recording. This predictor is the prototype of a model for reaDream, a brain implant which detects and decodes sleeping brain activation patterns.

Data used in Thesis: _A library of human electrocorticographic data and analyses_ (https://exhibits.stanford.edu/data/catalog/zk881ps0522)

**Important:** currently, **codes are not described, not clean and may not work in some cases.** A rewritten version is incoming in a few days which will be much easier to understand and configurate to reproduce the results.

Traning can be done either on CPU or GPU (with CUDA 10.2).

Essential packages:
* Numpy 1.20.3
* Scipy 1.6.2
* Sklearn 0.24.1
* PyTorch 1.10.1

Additional packages (can be removed if their code is also removed, does not affect the models):

* Matplotlib 3.3.4 (for plotting)
* jsonpickle 2.0.0 (for logging parameters)
* tqdm 4.62.1 (for pretty progress bars)
