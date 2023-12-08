# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import dirichelet
from sklearn.model_selection import train_test_split


class DiricheletClassifier:
    def __init__(self):
        self.model = None

    def train(self, corpus):
        ## should return a model
        return self.model

    def test(self):
        pass


def main():
    df = pd.read_csv("./events")
    

if __name__ == "__main__":
    main()
