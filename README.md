# Evolution of cooperation mediated by complex social traits
This repository contains code used to model the evolution of complex social traits. Here, a complex social trait is differentiated from a "simple" one by the number of parameters determining its social behavior. Classical models of evolving social interaction (such as those presented by Hamilton, Price, Axelrod, etc.) considered social behavior as captured by a single variable representing an individual's level of altruism. This model draws on work of Le Nagard et al. (2011) to consider the evolution of a complex trait, where an individual's willingness to cooperate is represented by several individually evolving traits. We use a feedforward neural network to capture this complexity, with individual node weights and biases representing the evolving components of the overall social trait. 

This directory contains two folders:

* **Python** : The Python folder contains prototyping code, which implemented the basics of the model (social interaction, evolution, and population structure). It also contains the basic code necessary to show that the model is congruent with earlier theory, as the results of classical predictions in population genetic theory will be recovered if the code is run with networks containing only a single node. It contains the following files:
  - **NetworkGame.py** : Basic prototyping code, contains necessary functions to initiate a population of neural networks, allow them to engage in social interaction, and subject them to evolution. It outputs a .H5 file containing simulation data
  - **Network_fig.py** : Supplemental file used to create figures
  - **selection_analysis.py** : Code to analyses the output of NetworkGame.py and show that it matches theoretical predictions
 
* **Julia** : The Julia folder contains high-performance code that maintains the functionality of the earlier Python code. Once compiled, it is capable of producing simulation outputs at ~100x the speed of the Python implementation. It also implements Distributed.jl, allowing the code to be ran on multiple cores or HPC clusters. 
