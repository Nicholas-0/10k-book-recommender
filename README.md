# 10k Book Recommender
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/pull/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)



This project is a book recommender system which makes use of machine learning to determine what books a user
would like based on those that they have previously rated. It is written in Python and created with various libraries such
as Numpy and TensorFlow, a deep learning framework which is used to develop and train machine learning models. Our 
data is taken from the [goodbooks-10k dataset on Kaggle](https://www.kaggle.com/zygmunt/goodbooks-10k).

![Top 10 books recommended to a user](Images/Top-10-books-recommended-for-a-user.png)
*Top 10 books recommended to a user*

## Dependencies
The following libraries are required to run `book_recommender_main.py`:

* [Python](https://www.python.org/) 3.7.7 or higher
* [TensorFlow](https://www.tensorflow.org/install) 2.1.0
* [Scikit-learn](https://scikit-learn.org/stable/install.html) 0.23.2
* [Numpy](https://numpy.org/install/) 1.19.1
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) 1.1.1 or higher
* **Other dependencies:** CUDA, CuDNN ([GPU support | TensorFlow](https://www.tensorflow.org/install/gpu)) - for training deep neural net on GPU

The easiest way to install all these libraries is to install them as part of the
[Anaconda](https://docs.anaconda.com/anaconda/install/) distribution,
a free and open-source Python distribution for scientific computing
which aims to simplify package management. I highly recommend this
as Anaconda uses the idea of environments so as to isolate
different libraries and versions. As an alternative, you can install
them from source.

I have also created a notebook `book_recommender_main.ipynb` in this repository which can
be run on Google Colab without the need to install all the libraries.

## Running the code
### Using Google Colab
I recommended running the code using `book_recommender_main.ipynb`. This way you
don't have to install all the libraries listed below as they are
already included in Colab. It also provides a free GPU which makes
training much faster if you don't have a GPU on your computer.

Note: You will need Git on your machine for it to work. You can install
Git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### Using an Integrated Development Environment (IDE)
Alternatively, you can run the code in `book_recommender_main.py` using
an IDE.


## Contributing
You are welcome to contribute to this project. Feel free to submit issues and enhancement requests.
### Pull Requests
1. **Fork** the repo on Github
2. **Clone** the project to your own machine
3. **Commit** changes to your own branch
4. **Push** your work back up to your fork
5. Submit a **Pull request** so that we can review your changes

## License
[MIT](https://choosealicense.com/licenses/mit/)



