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
import signal
from threading import Thread, Lock

from queue import Queue

# third party modules
import numpy as np

import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# original modules
from ..hp.hp import HP


class Experiment:
    """
    Class for Experiment using Basyan Optimization.

    Attributes:
        num_initial_iteration (int): the number of initial iterations before sampling based on baysian optimization.
        num_max_iteration (int): the number of iterations containing initial iterations.
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """

    # attributes
    num_epoch = None
    optimizer = None
    model = None
    loss_function = None
    epoch_id = None
    hyperparams = None
    train_dataset = None
    val_dataset = None
    val_frequence = None


    def __init__(self, hp_file_path, trainer):
        """関数の説明タイトル
        Initizer.

        Args:
            hp (:obj:`引数の型`, optional): The hyperparams of experiment that contain training parameter.
            trainer (:obj:`引数の型`, optional): The trainning instance.
        Returns:
            戻り値の型: 戻り値の説明 (例 : True なら成功, False なら失敗.)
        Raises:
            例外の名前: 例外の説明 (例 : 引数が指定されていない場合に発生 )

        Yields:
            戻り値の型: 戻り値についての説明

        Examples:

            # >>> from bo.trainer.trainer import SampleTrainer
            # >>> from hp.hp import get_hp

            print_test ("test", "message")
            test message

        Note:
            注意事項などを記載

        """

        # set basic paramter for experiment
        self.hp = HP(file_path=hp_file_path)
        self.experiment_name = self.hp.get("experiment.experiment_name")
        self.num_initial_iteration = self.hp.get("experiment.experiment_name")
        self.gpus = self.hp.get("experiment.experiment_name")
        self.num_gpus = len(self.gpus)

        # hyper param in this queues
        self.q_for_pending_tasks = Queue()

        # procces running on each gpus
        self.task_running = [None for i in range(self.num_gpus)]


        # init gaussian process

        self.trainer = trainer

        self.trials = []


    def run(self, print_console=True):
        """関数の説明タイトル
        The method to start experiment.

        Args:
            print_console (bool, optional): If True, console_log is printed while experiment.
        Returns:
            bool: If experiment has completed correctly, return True , else return False (例 : True なら成功, False なら失敗.)
        Raises:
            例外の名前: 例外の説明 (例 : 引数が指定されていない場合に発生 )

        Yields:
            戻り値の型: 戻り値についての説明

        Examples:

            関数の使い方について記載
            >>> print_test ("test", "message")
            test message

        Note:
            注意事項などを記載

        """
        # Done initial iters

        iter_id = 0

        # generate initial parameters and done each iter
        initial_params_for_trainer, initial_params_for_gp = self.hp.generate_random_sample(self.num_initial_iterations)

        for i in range(self.num_initial_iterations):
            train_x, hp_fix = self.hp.random_init(with_hp=True)

        for i in range(self.num_epoch):
            # excecut an epoch training
            self.train_epoch(i + 1)

            # if epoch is for validation, excecute validation.
            if (i + 1) % self.val_frequence == 0:
                self.val(i + 1)


    def put_one_iter(self, param, iter_id, gpu_id):
        trainer = self.Trainer(param, iter_id, gpu_id)

        if self.task_running[gpu_id] is not None:
            raise Exception()

        self.task_running[gpu_id] = trainer = self.Trainer(param, iter_id, gpu_id)

    def has_task_finished(self):
        has_all_iter_finished = True
        has_all_tasks_started = self.q_for_pending_tasks.qsize() == 0

        if has_all_tasks_started:
            for proc in self.task_running:
                if proc is None:
                    pass
                else:
                    if proc.is_alive():
                        have_all_iter_finished = False
                    else:
                        # get results

                        self.results.append()

            else:
                return True

        else:
            have_all_iter_finished = False
            for proc in self.task_running:
                if proc.is_alive():
                    pass
                else:
                    return False

            return has_all_iter_finished

        if has_all_tasks_started and are_all_gpus_wating:
            # all iter has finished!
            return True

        elif has_all_tasks_started:
            return False

        elif are_all_gpus_working:
            # all gpus are running so can not a put new iter
            return False

        elif are_all_gpus_working:
            # put a new iter on waiting gpu
            return False

        else:
            rasie Exception()

        # if true, finished, elif false,  running
        if self.task_running.count(None) == len(self.task_running.count(None)):
            pass
        for gpu_id, proc in enumerate(self.task_running):
            if not proc.is_alive:
                proc = None
                return gpu_id

        else:
            return None

        return id

    def logging(self, log, type):
        pass


class ThreadManager:
    """
    Class for Experiment using Basyan Optimization.

    Attributes:
        num_initial_iteration (int): the number of initial iterations before sampling based on baysian optimization.
        num_max_iteration (int): the number of iterations containing initial iterations.
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """

    # attributes

    def __init__(self, devices, trainer):
        """関数の説明タイトル
        Initizer.

        Args:
            hp (:obj:`引数の型`, optional): The hyperparams of experiment that contain training parameter.
            trainer (:obj:`引数の型`, optional): The trainning instance.
        Returns:
            戻り値の型: 戻り値の説明 (例 : True なら成功, False なら失敗.)
        Raises:
            例外の名前: 例外の説明 (例 : 引数が指定されていない場合に発生 )

        Yields:
            戻り値の型: 戻り値についての説明

        Examples:

            # >>> from bo.trainer.trainer import SampleTrainer
            # >>> from hp.hp import get_hp

            print_test ("test", "message")
            test message

        Note:
            注意事項などを記載

        """

        # set basic paramter for experiment
        self.devices = devices
        self.device_dict = {device: {"task": None, "event": } for device in self.devices}
        self.q_task_pending = Queue()
        self.q_task_finished = Queue()

        self.lock = Lock()

    def add_task(self, task):
        """関数の説明タイトル
        The method to start experiment.

        Args:
            task (bool, optional): If True, console_log is printed while experiment.
        Returns:
            bool: If experiment has completed correctly, return True , else return False (例 : True なら成功, False なら失敗.)
        Raises:
            例外の名前: 例外の説明 (例 : 引数が指定されていない場合に発生 )

        Yields:
            戻り値の型: 戻り値についての説明

        Examples:

            関数の使い方について記載
        Note:
            注意事項などを記載

        """
        self.lock.acquire()
        # Done initial iters
        # check if there are free device
        for key, value in self.device_dict.items():
            # check if value is None
            if value is None:
                task.set_device(key)
                task.setDaemon(key)
                value = task
                break

        # if all device is busy, task will pending
        else:
            self.q_task_pending.put(task)

        self.lock.release()

    def manage_event(self):
        

    def put_one_iter(self, param, iter_id, gpu_id):
        trainer = self.Trainer(param, iter_id, gpu_id)

        if self.task_running[gpu_id] is not None:
            raise Exception()

        self.task_running[gpu_id] = trainer = self.Trainer(param, iter_id, gpu_id)

    def has_task_finished(self):
        has_all_iter_finished = True
        has_all_tasks_started = self.q_for_pending_tasks.qsize() == 0

        if has_all_tasks_started:
            for proc in self.task_running:
                if proc is None:
                    pass
                else:
                    if proc.is_alive():
                        have_all_iter_finished = False
                    else:
                        # get results

                        self.results.append()

            else:
                return True

        else:
            have_all_iter_finished = False
            for proc in self.task_running:
                if proc.is_alive():
                    pass
                else:
                    return False

            return has_all_iter_finished

        if has_all_tasks_started and are_all_gpus_wating:
            # all iter has finished!
            return True

        elif has_all_tasks_started:
            return False

        elif are_all_gpus_working:
            # all gpus are running so can not a put new iter
            return False

        elif are_all_gpus_working:
            # put a new iter on waiting gpu
            return False

        else:
            rasie Exception()

        # if true, finished, elif false,  running
        if self.task_running.count(None) == len(self.task_running.count(None)):
            pass
        for gpu_id, proc in enumerate(self.task_running):
            if not proc.is_alive:
                proc = None
                return gpu_id

        else:
            return None

        return id

    def logging(self, log, type):
        pass

