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
from multiprocessing import Queue

from logging import getLogger

# third party modules
import numpy as np

import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# original modules
from ..hp.hp import HP


logger_master_name = "BO _Experiment_Log"
logger = getLogger(__name__)


