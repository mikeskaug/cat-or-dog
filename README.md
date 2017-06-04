# Is it a Cat or a Dog?

That's the question that this convolutional neural network is supposed to answer. It is my experiment with the classic [Dogs vs. Cats Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).

## Usage

**Dependencies**

The only Python dependencies not in the standard library are:

  * tensorflow
  * opencv
  * numpy

You can install all of the dependencies using [conda](https://conda.io/docs/index.html):

      $ conda env create -f environment.yml

**Training**

The training and testing data is available from the Kaggle challenge [data page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) and should be placed in a top level `data/` directory

Then you can train the network:

      $ python train.py

To visualize training metrics, run:

      $ python -m tensorflow.tensorboard --logdir='./output'

and browse to `localhost:6006`
