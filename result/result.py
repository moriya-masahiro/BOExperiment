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

# original modules


# Define const value
STATUS_PENDING = 0
STATUS_RUNNING = 1
STATUS_DONE = 2
STATUS_FAILED = 3


class Result:
    def __init__(self, hp, experiment_name, parent_name, trial_id):
        self.parent_name = parent_name
        self.experiment_name =experiment_name
        self.trial_id = trial_id

        self.experiment_name = f"{self.parent_name}_trial_{self.trial_id}"

        self.hp = hp
        self.train_log = TrainLog()

        self.lock = lock

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


class TrainLog:
    def __init__(self, hogoe):
        self.parent_name = parent_name
        self.trial_id = trial_id

        self.experiment_name = f"{self.parent_name}_trial_{self.trial_id}"

        self.hp = hp

        self.status = STATUS_PENDING

    def get_status(self):
        status_dict = {STATUS_PENDING: "pending",
                       STATUS_RUNNING: "running",
                       STATUS_DONE: "done",
                       STATUS_FAILED: "failed"}

        return status_dict[self.status]