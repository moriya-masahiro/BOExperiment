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
import os
import itertools

from pathlib import Path
from copy import deepcopy

# third party modules
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.distributions

# original modules
from ..utils.const import *


class DataStructure:
    def __init__(self, names, values, types, uses):
        if not (len(names) == len(values) == len(types) == len(uses)):
            raise Exception("nanka kaku!")

        self.all_values = []
        self.batch_size = None

        for _name, _value, _type, _use in zip(names, values, types, uses):
            self.all_values.append(_name)
            setattr(self, _name, DataElememt(_name, _value, _type, _use))

            if self.batch_size is None:
                self.batch_size = len(_value)

            elif self.batch_size != len(_value):
                raise Exception("nanka kaku!")

            else:
                pass

        self.index = 0
        self.specified = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __next__(self, item):
        if self.index == self.__len__():
            raise StopIteration()
        else:
            value = getattr(self, self.all_values[self.index])
            self.index += 1

            return value

    def __len__(self):
        return self.all_values

    def specify(self, name):
        if name in self.all_values:
            self.specified = name
        else:
            raise Exception("nanka kaku!")


class DataElememt:
    def __init__(self, _name, _value, _type, _use):
        self.name = _name
        self.value = _value
        self.type = _type
        self.use = _use

    def get_value(self):
        return self.value

    def get_name(self):
        return self.name

    def get_type(self):
        return self.type

    def get_use(self):
        return self.use
