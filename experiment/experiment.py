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
import signal
import time
from threading import Thread, Lock, Event

from queue import Queue
from copy import deepcopy

# third party modules
import numpy as np

import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import UpperConfidenceBound

from botorch.optim import optimize_acqf

# original modules
from hp.hp import HP
from utils.const import *


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

    def __init__(self, hp, TrainerClass):
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
        self.hp = hp
        self.experiment_name = self.hp.get("experiment.experiment_name")
        self.num_initial_iteration = self.hp.get("experiment.experiment_name")
        self.gpus = self.hp.get("experiment.devices")
        print(self.gpus)
        self.num_gpus = len(self.gpus)

        self.thread_manager = ThreadManager(self.gpus)

        # init gaussian process

        self.TrainerClass = TrainerClass

        self.results = []
        self.train_x = []
        self.train_y = []

        self.best_y = torch.tensor([-100])

        self.model = None


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

        # generate initial parameters and done each
        for i in range(int(self.hp.experiment.num_initial_iterations)):

            # generate hp
            new_train_x, hp_fix = self.hp.generate_random_sample(1, with_hp=True)

            hp_fix.print_all()

            # generate trainer and add to
            trainer = self.TrainerClass(str(self.hp.experiment.experiment_name),
                                        iter_id,
                                        hp_fix)

            self.thread_manager.add_task(trainer, new_train_x)

            iter_id += 1

        # while iter_id < int(self.hp.experiment.num_iterations):
        #    self.model
        #    print("hoge")
        self.thread_manager.run()

        while(True):
            print(f"length of results is {len(self.results)}")
            if self.thread_manager.q_task_finished.qsize() > 0:
                self.results.append(self.thread_manager.get_finished_task())
                print(self.results[-1].trainer.result.results[0].get_value())

                new_train_y = torch.tensor([self.results[-1].trainer.result.results[0].get_value()])
                new_train_x = self.results[-1].train_x

                if new_train_y > self.best_y:
                    self.best_y = new_train_y
                    self.best_x = new_train_x

                # add new observation
                if isinstance(self.train_x, list):
                    self.train_x = new_train_x.unsqueeze(0)
                    self.train_y = new_train_y.unsqueeze(0)

                else:
                    self.train_x = torch.cat([self.train_x, new_train_x.unsqueeze(0)])
                    self.train_y = torch.cat([self.train_y, new_train_y.unsqueeze(0)])

                print(self.train_x.size())
                print(self.train_y.size())

                if iter_id < int(self.hp.experiment.num_iterations):
                    gp = SingleTaskGP(self.train_x, self.train_y)
                    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                    fit_gpytorch_model(mll)
                    UCB = UpperConfidenceBound(gp, beta=0.1)

                    bounds = self.hp.get_bounds()
                    candidate, acq_value = optimize_acqf(
                        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
                    )

                    print(candidate, acq_value)

                    new_train_x = candidate[0]
                    hp_fix = self.hp.get_const_hp_from_tensor(new_train_x)

                    trainer = self.TrainerClass(str(self.hp.experiment.experiment_name),
                                                iter_id,
                                                hp_fix)

                    self.thread_manager.add_task(trainer, new_train_x)
                    iter_id += 1

                elif self.thread_manager.is_finished():
                    break

            print("sleep")
            time.sleep(0.1)

    def logging(self, log, type):
        pass

    def get_best_result(self):
        return self.best_x, self.best_y


class ThreadManager:
    """
    Class for Experiment using Basyan Optimization.

    Attributes:
        num_initial_iteration (int): the number of initial iterations before sampling based on baysian optimization.
        num_max_iteration (int): the number of iterations containing initial iterations.
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """

    # attributes

    def __init__(self, devices):
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

        # set basic parameter for experiment
        self.devices = devices
        self.device_dict = {device: False for device in self.devices}
        self.q_task_pending = Queue()
        self.q_task_finished = Queue()
        self.list_task_running = []

        self.manager_thread = self.__launch_manager_thread()
        self.manager_thread.setDaemon(True)

        self.event = Event()
        self.lock = Lock()

    def run(self):
        self.manager_thread.start()

    def __launch_manager_thread(self):
        manager_thread = Thread(target=self.__manager_func)

        return manager_thread

    def __manager_func(self):
        while True:
            # set new task to free device
            for i, task in enumerate(self.list_task_running):
                if not task.is_alive():
                    self.q_task_finished.put(task)
                    self.list_task_running.pop(i)
                    self.device_dict[task.device] = False
                    break

            time.sleep(1)

    def __launch_task_thread(self, trainer, new_train_x, device):
        class TaskThread(Thread):
            def __init__(self, _trainer, train_x, _device, event, lock):
                super(TaskThread, self).__init__()
                self.trainer = _trainer
                self.device = _device
                self.trainer.set_device(_device)
                self.event = event
                self.lock = lock
                self.train_x = train_x

            def run(self):
                self.trainer.run()

            def get_result(self):
                return self.trainer.get_result()

            def get_status(self):
                return self.trainer.get_status()

        return TaskThread(trainer, new_train_x, device, self.event, self.lock)

    def add_task(self, _trainer, new_train_x):
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
        # Done initial iters
        # check if there are free device
        for key, value in self.device_dict.items():
            # check if value is None
            if not value:
                task_thread = self.__launch_task_thread(_trainer, new_train_x, key)
                print(self.device_dict)
                task_thread.setDaemon(True)
                self.device_dict[key] = True

                self.list_task_running.append(task_thread)
                self.list_task_running[-1].start()
                print(self.device_dict)
                break

        # if all device is busy, task will pending
        else:
            self.q_task_pending.put([_trainer, new_train_x])

    def __len__(self):
        return self.q_task_finished.qsize()

    def get_results(self):
        results = []
        while self.q_task_finished.not_empty:
            result = self.q_task_finished.get().get_result()
            results.append(result)

        return results

    def get_result(self):
        result = self.q_task_pending.get()

        return result

    def get_finished_task(self):
        task = self.q_task_finished.get()
        return task

    def logging(self, log, __type):
        pass

    def is_finished(self):
        if self.q_task_pending.not_empty:
            return True

        for key, value in self.device_dict.items():
            # check if value is None
            if value:
                return False

        return True

