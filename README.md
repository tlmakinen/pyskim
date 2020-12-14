# pyskim
Repository for [SKIM Bayesian sparse regression](https://arxiv.org/abs/1905.06501) in Jax and NumPyro. Code adapted from [this paper]
(https://arxiv.org/abs/1905.06501) and [this tutorial](http://num.pyro.ai/en/latest/examples/sparse_regression.html) with major modifications to plotting routines
and hyperparameter optimization. More verbose explanation coming soon. Check out this [Google Colab tutorial]
(https://colab.research.google.com/drive/1hZPsjLAQGobymd1Hc2qISJXMXscYGsU3#scrollTo=iywDKXqeE6oj) highlighting the SKIM kernel technique to find predictors for COVID-19 mortality rates by country on data circa 
April 2020.

# installation
To install SKIM and its dependencies in your Python environment, first clone the repository:

`git clone https://github.com/tlmakinen/pyskim.git`

then, 
`cd ../pyskim`
and install dependencies and the SKIM module via

`python3 setup.py install`

# usage
For an intuituve walkthrough and review of the math behind the kernel, see the above Colab Notebook. To see if your module is working properly on your OS, try
`python _test/test.py`
for a quick test of the module. Note for Windows subsystem for Linux users: the testing function should display a sample corner plot for the test covariates through `matplotlib`. Be sure to have X window forwarding enabled !
