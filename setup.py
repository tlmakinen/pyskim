
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SKIM",
    version="0.10",
    author="Lucas Makinen",
    author_email="tmakinen@princeton.edu",
    description="Learning sparse interactions through fast Bayesian Hierarchical \
    Sampling via the kernel trick ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tlmakinen/pyskim.git",
    packages=["SKIM"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
    install_requires=[
          "jax>=0.2.3",
          "numpyro>=0.4.1",
          "tqdm>=4.31.0",
          "numpy>=1.16.0",
          "scipy>=1.4.1",
          "corner",
          "matplotlib"],
)
