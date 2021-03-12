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

    required_params = ("train.x0",
                       "train.x1",
                       "train.x2",
                       "train.x3",
                       "train.x4",
                       "train.x5")

    def __init__(self, root, parent_name, trial_id, hp, embedding_id=None, device=None, logger=None):
        self.root = root
        self.status = STATUS_PENDING
        self.parent_name = parent_name
        self.trial_id = trial_id
        self.device = device
        self.embedding_id = embedding_id
        self.logger = logger

        self.trial_name = f"{self.parent_name}_trial_{self.trial_id}"

        self.hp = hp

        self.lock = Lock()

        self.hartmann = Hartmann(negate=True)

        self.result = Result(self.hp, self.trial_name, self.parent_name, self.trial_id)

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
        self.status = status
        self.lock.release()

    def get_name(self):
        self.lock.acquire()
        name = self.trial_name
        self.lock.release()

        return name

    def get_result(self):
        self.lock.acquire()
        result = self.result
        self.lock.release()

        return result

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

        # set results
        self.result.set_result(INDEX_TYPE_NORMAL_CONTINUOUS, "hartmann_output", output.sum().item(), 0, 0, PHASE_TEST)

        # set status
        if self.logger is not None:
            self.logger.info(f"The trial (id={self.trial_id}) has finished.")
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
