#!/usr/bin/python
# -*- coding: utf-8 -*-

"""モジュールの説明タイトル

     * ソースコードの一番始めに記載すること
     * importより前に記載する

Todo:
   TODOリストを記載
    * conf.pyの``sphinx.ext.todo`` を有効にしないと使用できない
    * conf.pyの``todo_include_todos = True``にしないと表示されない

"""

# standard modules
import sys

from statistics import mean

sys.path.append('../')

import time

# third party modules
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

import torch

# original modules
from utils.const import *


def str2bool(s):
    return s.lower() in ('true', '1')


def inverse_lookup(d, x):
    for k, v in d.items():
        if x == v:
            return k


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval


class EarlyStopper:
    best = None
    model = LinearRegression()
    observation = []

    def __init__(self, probability_thres=0.5, idling_period=10, observation_period=10, grace_period=10, logger=None):
        self.probability_thres = probability_thres
        self.idling_period = idling_period
        self.observation_period = observation_period
        self.grace_period = grace_period
        self.logger = logger

    def is_converged(self, new_point=None):
        if new_point:
            self.observation.append(new_point)
            self.best = max(self.observation)

        if len(self.observation) < self.idling_period:
            return False

        # fit
        Y = np.array(self.observation[-self.observation_period:])
        X = np.array(range(len(Y)))
        self.model.fit(X, Y)

        coef, intercept = self.model.coef, self.model.intercept_

        scale = np.sqrt(mean([(x * coef + intercept - y) ** 2 for x, y in (zip(X.tolist(), Y.tolist()))]))

        # calc probability of exceed
        probability_of_exceed = 1
        for i in range(self.grace_period):
            probability_of_exceed *= (1 - norm.cdf(x=self.best, loc=coef * (i + len(X)), scale=scale))

        if probability_of_exceed > self.probability_thres:
            return False
        else:
            return True


class RandomMatrixManager:
    def __init__(self, dim_feature, num_matrices=1, max_dim_embedding=None, min_dim_embedding=None, bo_type=None):
        self.dim_feature = dim_feature
        self.num_matrices = num_matrices
        self._current = 0

        if max_dim_embedding is None or min_dim_embedding is None:
            self.max_dim_embedding = self.dim_feature
            self.min_dim_embedding = self.dim_feature // 3
        else:
            self.max_dim_embedding = max_dim_embedding
            self.min_dim_embedding = min_dim_embedding

        self.bo_type = bo_type

        self.shuffle_list = list(range(self.num_matrices))
        self.shuffle_id = 0

        self.matrix_dict = {}
        for i in range(self.num_matrices):
            embedding_matrix = self.generate_matrix()
            self.matrix_dict[i] = {"embedding_matrix": embedding_matrix,
                                   "dim_embedding": int(embedding_matrix.size(0)),
                                   "indicator": 0.}

        self.active_index = list(range(self.num_matrices))

        self.reduction = 0

    def __len__(self):
        return self.num_matrices

    def generate_matrix(self, dim_embedding=None):
        if dim_embedding is None:
            dim_embedding = np.random.randint(self.min_dim_embedding, self.max_dim_embedding + 1)

        while True:
            # matrix = torch.rand(dim_embedding, self.dim_feature) - 0.5
            matrix = torch.normal(0, 1, size=(dim_embedding, self.dim_feature))

            matrix2 = torch.mm(torch.t(matrix), matrix)

            if torch.matrix_rank(matrix2) == torch.tensor(dim_embedding):
                # return matrix / matrix2.det()
                return matrix
            else:
                pass
                # print(f"rank: {torch.matrix_rank(matrix2)}")
                # print(f"dim_feature: {torch.tensor(self.dim_feature)}")
                # print(f"dim_embedding: {dim_embedding}")
                # print(f"size: {matrix2.size()}")
                # print(f"matrix2: {matrix2}")
                # print(f"matrix: {matrix}")

    def update_indicator(self, index, value):
        self.matrix_dict[index]["indicator"] = max(self.matrix_dict[index]["indicator"], value)

    def update_matrices_except_best_one(self, best_index):
        # update actives
        for i in self.active_index:
            if i != best_index:
                new_dim_embedding = np.random.choice(list(range(self.min_dim_embedding, self.max_dim_embedding + 1)))

                new_embedding_matrix = self.generate_matrix(dim_embedding=new_dim_embedding)
                new_active_index = len(self.matrix_dict)

                self.matrix_dict[new_active_index] = {"embedding_matrix": new_embedding_matrix,
                                                      "dim_embedding": new_dim_embedding,
                                                      "indicator": 0.}

                # update actives
                self.active_index[self.active_index.index(i)] = new_active_index

        print([self.matrix_dict[active]["dim_embedding"] for active in self.active_index])

    def update_matrices(self):
        min_indicator = 1e16
        min_index = 0

        # choose an update matrix
        for index in self.active_index:
            if self.matrix_dict[index]["indicator"] < min_indicator:
                min_index = index
                min_indicator = self.matrix_dict[index]["indicator"]

        # update
        # self.dim_dict[self.matrix_dict[min_index]["dim_embedding"]].append(min_indicator)

        # generate weights list
        weights_list = [[] for i in range(self.min_dim_embedding, self.max_dim_embedding + 1)]
        for value in self.matrix_dict.values():
            dim = value["dim_embedding"]
            indicator = value["indicator"]
            weights_list[self.dim2index(dim)].append(indicator)

        weights = [max(weight_list) if len(weight_list) != 0 else -1. for weight_list in weights_list]

        for index in range(len(weights)):
            if weights[index] == -1:
                weights[index] = max(weights)

        for index in range(len(weights)):
            weights[index] = weights[index] ** 2 + 1e-16

        weights = [(weight) / (sum(weights)) for weight in weights]

        hoge = {i: j for i, j in zip(list(range(self.min_dim_embedding, self.max_dim_embedding + 1)), weights)}

        print(f"weights: {hoge}")

        # generate new matrix
        new_dim_embedding = np.random.choice(list(range(self.min_dim_embedding, self.max_dim_embedding + 1)),
                                             p=weights)

        new_embedding_matrix = self.generate_matrix(dim_embedding=new_dim_embedding)
        new_active_index = len(self.matrix_dict)

        self.matrix_dict[new_active_index] = {"embedding_matrix": new_embedding_matrix,
                                              "dim_embedding": new_dim_embedding,
                                              "indicator": 0.}

        # update actives
        self.active_index[self.active_index.index(min_index)] = new_active_index

    def index2dim(self, index):
        return index + self.min_dim_embedding

    def dim2index(self, dim):
        return dim - self.min_dim_embedding

    def get_embedding_matrix_random(self):
        if self.shuffle_id == self.num_matrices:
            np.random.shuffle(self.shuffle_list)
            self.shuffle_id = 0

        index = self.active_index[self.shuffle_list[self.shuffle_id]]

        self.shuffle_id += 1
        # index = np.random.choice(self.active_index)

        return self.matrix_dict[index]["embedding_matrix"], index

    def get_embedding_matrix(self, index):
        return self.matrix_dict[index]["embedding_matrix"]

    def __iter__(self):
        return self

    def __next__(self):
        self._current += 1
        if self._current >= self.num_matrices:
            self._current = 0
            raise StopIteration

        else:
            return self.matrix_dict[self._current]["embedding_matrix"]
