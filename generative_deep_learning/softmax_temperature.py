"""
Reweighting a probability distribution to a different temperature
"""
import numpy as np


# original_distribution is a 1D Numpy array of probability values
# that must sum to 1. temperature is a factor quantifying the entropy
# of the output distribution.
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)

    # Returns a reweighted version of the original distribution.
    # The sum of the distribution may no longer be 1,so you divide
    # it by its sum to obtain the new distribution.
    return distribution / np.sum(distribution)
