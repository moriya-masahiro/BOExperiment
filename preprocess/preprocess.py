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
sys.path.append('../')
import os
import itertools

from pathlib import Path
from copy import deepcopy

# third party modules
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

import torchvision

# original modules
from utils.const import *

import albumentations as albu


class RandomGammaTransform(nn.module):
    def __init__(self, dist_type, scale=None, max_value=None, min_value=None):
        if dist_type not in ALL_DIST_TYPE:
            raise Exception("nanka kaku!")

        self.dist_type = dist_type
        self.scale = scale
        self.max_value = max_value
        self.min_value = min_value

        self.transform_label = False

        if self.dist_type == TYPE_DIST_NOROMAL:
            self.dist = torch.distributions.normal.Normal(torch.tensor([1.]),
                                                     torch.tensor([self.scale]))

        elif self.dist_type == TYPE_DIST_UNIFORM:
            self.dist = torch.distributions.uniform.Uniform(torch.tensor([self.max_value]),
                                                       torch.tensor([self.min_value]))

        else:
            raise Exception("nanka kaku!")

    def forward(self, x, is_active=None):
        batch_size = x.batch_size
        if is_active is not None:
            if len(is_active) == batch_size:
                is_active = torch.tensor(is_active * batch_size)
            else:
                raise Exception("nanka kaku!")
        else:
            is_active = torch.tensor([1] * batch_size)

        with torch.no_grad():
            for key, value in x.items():
                # set device of is_active tensor to its of x
                is_active = is_active.to(value.deivce)

                # transform
                if value.use == TYPE_DATA_USE_INPUT and value.type == TYPE_DATA_IMAGE:
                    value.value[is_active] = value.value[is_active] ** \
                                             self.dist.sample(value.data.size(0))[is_active]

        return x


class RandomAffineTransform(nn.module):
    def __init__(self, dist_type, scale=None, max_value=None, min_value=None):
        pass

    def forward(selfself, x, is_active=None):
        pass


class RandomActivation(nn.Module):
    def __init__(self, preprocess, activation_rate):
        self.preprocess = preprocess
        self.activation_rate =activation_rate
        self.dist = torch.distributions.bernoulli.Bernoulli(torch.tensor([activation_rate]))

    def forward(self, x):
        batch_size = x.batch_size
        is_active = self.dist.sample(batch_size)

        with torch.no_grad():
            x = self.preprocess(x, is_active)

        return x


class RandomChoice(nn.Module):
    def __init__(self, preprocesses, weights=None):
        self.preprocesses = preprocesses

        if weights is None:
            self.weights = [1./len(self.preprocesses) for i in range(self.preprocesses)]
        elif sum(weights) > 0 and len(weights) == len(self.preprocesses):
            self.weights = [weight / sum(weights) for weight in weights]
        else:
            raise Exception("nanka kaku!")

    def forward(self, x):
        batch_size = x.batch_size
        with torch.no_grad():
            for i in range(batch_size):
                preprocess_id = np.random.choice(len(self.preprocesses), 1, p=self.weights)

                is_active = torch.nn.functional.one_hot(torch.tensor([i]), num_classes=len(batch_size))

                x = self.preprocesses[preprocess_id](x, is_active)

        return x


class MyToTensor(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


def get_preprocess(hp):
    transform_list = []

    hp.get_const("train.preprocess.")



