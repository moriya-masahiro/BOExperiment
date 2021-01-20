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

# original modules
from result.result import Result
from utils.const import *


class Trainer:
    def __init__(self, parent_name, trial_id, hp):
        self.status = STATUS_PENDING
        self.parent_name = parent_name
        self.trial_id = trial_id

        self.experiment_name = f"{self.parent_name}_trial_{self.trial_id}"

        self.hp = hp

        self.lock = Lock()

        self.result = Result(self.hp, self.experiment_name, self.parent_name, self.trial_id)

    def get_status(self):
        self.lock.acquire()
        status_dict = {STATUS_PENDING: "pending",
                       STATUS_RUNNING: "running",
                       STATUS_DONE: "done",
                       STATUS_FAILED: "failed"}

        status = status_dict[self.status].results[0]
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

    def get_result(self):
        self.lock.acquire()
        result = self.result
        self.lock.release()

        return result

    def dump_to_file(self, path):
        pass

