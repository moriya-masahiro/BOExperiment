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

from threading import Lock

# third party modules
import numpy as np
import yaml

import torch

from botorch.test_functions import Hartmann

# original modules
from result.result import Result
from utils.const import *

from trainer.trainer import Trainer


status_dict = {STATUS_PENDING: "pending",
                       STATUS_RUNNING: "running",
                       STATUS_DONE: "done",
                       STATUS_FAILED: "failed"}


class Hartmann6Trainer(Trainer):
    def __init__(self, parent_name, trial_id, hp, device=None):
        self.status = STATUS_PENDING
        self.parent_name = parent_name
        self.trial_id = trial_id
        self.device = device

        self.experiment_name = f"{self.parent_name}_trial_{self.trial_id}"

        self.hp = hp

        self.lock = Lock()

        self.hartmann = Hartmann(negate=True)

        self.result = Result(self.hp, self.experiment_name, self.parent_name, self.trial_id)

    def get_status(self):
        self.lock.acquire()


        # status = status_dict[self.status]
        self.lock.release()
        # return status
        return self.status

    def set_status(self, status):
        if status not in [STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED]:
            raise Exception("nanka kaku!")

        self.lock.acquire()
        print(f"set status: {status_dict[status]}")
        self.status = status
        self.lock.release()

    def get_name(self):
        self.lock.acquire()
        name = self.experiment_name
        self.lock.release()

        return name

    def get_result(self):
        self.lock.acquire()
        name = self.experiment_name
        self.lock.release()

        return name

    def dump_to_file(self, path):
        pass

    def run(self):
        # set status
        self.set_status(STATUS_RUNNING)

        # create input tensor

        x = torch.tensor([[self.hp.train.x0.get_value(),
                           self.hp.train.x1.get_value(),
                           self.hp.train.x2.get_value(),
                           self.hp.train.x3.get_value(),
                           self.hp.train.x4.get_value(),
                           self.hp.train.x5.get_value()]], dtype=torch.float32)

        output = self.hartmann(x) * (self.__outcome_constraint() <= 0).type_as(x)

        print(f"the output of hartmann is {output}")

        # set results
        self.result.set_result(INDEX_TYPE_NORMAL_CONTINUOUS, "hartmann_output", output.sum().item(), 0, PHASE_TEST)

        # set status
        self.set_status(STATUS_DONE)

    def __outcome_constraint(self):
        x = torch.tensor([[self.hp.train.x0.get_value(),
                           self.hp.train.x1.get_value(),
                           self.hp.train.x2.get_value(),
                           self.hp.train.x3.get_value(),
                           self.hp.train.x4.get_value(),
                           self.hp.train.x5.get_value()]], dtype=torch.float32)

        return x.sum(dim=-1) - 3

    def set_device(self, device):
        self.device = device
