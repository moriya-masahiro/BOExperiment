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

from threading import Lock

from pathlib import Path
from copy import deepcopy

# third party modules
import numpy as np
import yaml

import torch

# original modules
from utils.const import *
from utils.utils import *


class Result:
    def __init__(self, hp, experiment_name, parent_name, trial_id):
        self.parent_name = parent_name
        self.experiment_name = experiment_name
        self.trial_id = trial_id

        self.experiment_name = f"{self.parent_name}_trial_{self.trial_id}"

        self.hp = hp
        self.results = []

        self.lock = Lock()

        self.status = STATUS_PENDING

    def get_status(self):
        self.lock.acquire()
        status_dict = {STATUS_PENDING: "pending",
                       STATUS_RUNNING: "running",
                       STATUS_DONE: "done",
                       STATUS_FAILED: "failed"}

        status = status_dict[self.status]
        self.lock.release()
        return status

    def set_status(self, status):
        if status not in [STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED]:
            raise Exception("nanka kaku!")

        self.lock.acquire()
        self.status = status
        self.lock.release()

    def get_name(self):
        self.lock.acquire()
        name = self.experiment_name
        self.lock.release()

        return name

    def dump_to_file(self, path):
        pass

    def set_result(self, value_type, name, value, step, phase):
        self.results.append(EvalIndex(value_type, name, value, step, phase=phase))


class EvalIndex:
    def __init__(self, value_type, name, value, step, phase=PHASE_TRAIN):
        self.value_type = value_type
        self.value = value
        self.name = name
        self.step = step
        self.phase = phase

    def get_status(self):
        status_dict = {STATUS_PENDING: "pending",
                       STATUS_RUNNING: "running",
                       STATUS_DONE: "done",
                       STATUS_FAILED: "failed"}

        return status_dict[self.status]

    def get_value(self):
        return self.value

    def check_value(self):
        max_value = get_max(self.value_type)
        min_value = get_min(self.value_type)

    def __str__(self):
        return str(self.value)
