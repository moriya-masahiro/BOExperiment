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

from botorch.acquisition.analytic import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement, PosteriorMean

from botorch.optim import optimize_acqf

# original modules
from hp.hp import HP
from utils.const import *
from utils.utils import *


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
    train_dataset = None
    val_dataset = None

    def __init__(self, hp, TrainerClass, logger=None):
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
        self.hp = hp
        self.TrainerClass = TrainerClass
        self.logger = logger

        self.experiment_name = self.hp.get("experiment.experiment_name")
        self.num_initial_iteration = self.hp.get("experiment.num_initial_iterations")
        self.num_iterations = self.hp.get("experiment.num_iterations")
        self.gpus = self.hp.get("experiment.devices")
        self.acq_func_type = ALL_TYPE_ACQUISITION_FUNC[self.hp.get("experiment.acquisition_function_type")]
        self.bo_type = ALL_TYPE_REMBO[self.hp.get("experiment.bo_type")]
        self.root = self.hp.get("experiment.root_dir")

        self.matrix_manger = RandomMatrixManager(self.hp.get_dim(),
                                                 num_matrices=10,
                                                 bo_type=self.bo_type,
                                                 max_dim_embedding=6,
                                                 min_dim_embedding=4)

        if self.logger is not None:
            self.logger.info(f"Start the experiment:{self.experiment_name}")

        self.num_gpus = len(self.gpus)

        self.thread_manager = ThreadManager(self.gpus, self.num_iterations, logger=self.logger)

        self.results = []
        self.train_x = []
        self.train_y = []

        self.best_y = torch.tensor([-100])
        self.best_x = None

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

        if self.logger is not None and not self.hp.is_variable():
            self.logger.info(f"The hp has no variable so bayesian optimization will not be executed.")

        # if hp does not have variable, multi thread will not be launched.
        if not self.hp.is_variable():
            trainer = self.TrainerClass(self.root,
                                        str(self.hp.experiment.experiment_name),
                                        iter_id,
                                        self.hp,
                                        logger=self.logger,
                                        device=self.gpus[0])

            trainer.run()

            result = trainer.get_result()

            self.results.append(result)

            self.best_y = torch.tensor([result.get_best_result(phase=PHASE_TEST).get_value()])
            self.best_x = None

        # if hp has variable, multi thread will be launched.
        else:
            # generate initial parameters and done each
            for i in range(int(self.hp.experiment.num_initial_iterations)):

                # generate hp
                if self.logger is not None:
                    self.logger.info(f"Set the trial (id = {iter_id})")

                new_train_x, hp_fix = self.hp.generate_random_sample(1, with_hp=True)

                hp_fix.print_all()

                # generate trainer and add to
                trainer = self.TrainerClass(self.root,
                                            str(self.hp.experiment.experiment_name),
                                            iter_id,
                                            hp_fix,
                                            logger=self.logger)

                self.thread_manager.add_task(trainer, new_train_x)

                iter_id += 1

            self.thread_manager.run()

            iter_id_finished = 0
            while not self.thread_manager.is_finished():
                if self.thread_manager.q_task_finished.qsize() > 0:
                    iter_id_finished += 1
                    new_finished_task = self.thread_manager.get_finished_task()

                    new_result = new_finished_task.get_result()
                    self.results.append(new_result)

                    embedding_id = new_finished_task.trainer.embedding_id
                    if embedding_id is not None:
                        # update indicator
                        self.matrix_manger.update_indicator(embedding_id, new_result.get_best_result(phase=PHASE_TEST).get_value())
                        if (iter_id_finished + 1) % self.matrix_manger.num_matrices == 0:
                            self.matrix_manger.update_matrices()

                    # get result
                    new_train_y = torch.tensor([new_result.get_best_result(phase=PHASE_TEST).get_value()])
                    new_train_x = new_finished_task.train_x
                    trial_id = new_finished_task.trainer.trial_id

                    if self.logger is not None:
                        self.logger.info(f"Get the result trial (trial_id = {trial_id}). The result is {new_train_y}.")

                    # check and update best observation
                    if new_train_y > self.best_y:
                        self.best_y = new_train_y
                        self.best_x = new_train_x

                        if self.logger is not None:
                            self.logger.info(f"The best result has updated.")

                    # add new observation
                    if isinstance(self.train_x, list):
                        self.train_x = new_train_x.unsqueeze(0)
                        self.train_y = new_train_y.unsqueeze(0)

                    else:
                        self.train_x = torch.cat([self.train_x, new_train_x.unsqueeze(0)])
                        self.train_y = torch.cat([self.train_y, new_train_y.unsqueeze(0)])

                    if iter_id_finished < int(self.hp.experiment.num_initial_iterations):
                        # nothing to do
                        pass

                    elif iter_id < int(self.hp.experiment.num_iterations):
                        # type plane
                        if self.bo_type == TYPE_BO_PLANE:
                            acq_func = self.get_acq_func(self.train_x, self.train_y)

                            bounds = self.hp.get_bounds()
                            # if discrete parameter exists, ...
                            if self.hp.has_discrete():
                                best_candidate, best_acq_value = None, None
                                discrete_patterns = self.hp.discrete_patterns()
                                for pattern in discrete_patterns:
                                    candidate, acq_value = optimize_acqf(
                                        acq_func, bounds=bounds, fixed_features=pattern, q=1, num_restarts=5, raw_samples=20,
                                    )
                                    if best_acq_value is None:
                                        best_candidate, best_acq_value = candidate, acq_value
                                    elif acq_value > best_acq_value:
                                        best_candidate, best_acq_value = candidate, acq_value
                                    else:
                                        # best_candidate, best_acq_value = candidate, acq_value
                                        pass

                            # if no discrete parameter exists, ...
                            else:
                                best_candidate, best_acq_value = optimize_acqf(
                                    acq_func, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
                                )

                            if self.logger is not None:
                                self.logger.info(f"The candidate of next_x is {best_candidate}, acquisition is {best_acq_value}.")

                            new_train_x = best_candidate[0]
                            hp_fix = self.hp.get_const_hp_from_tensor(new_train_x)

                        # type rembo
                        elif self.bo_type == TYPE_BO_REMBO:
                            """embedding_matrix, embedding_id = self.matrix_manger.get_embedding_matrix()
                            train_x_embedded = torch.t(torch.mm(embedding_matrix, torch.t(self.train_x)))
                            bounds_embedded = self.hp.get_bounds(embedding_matrix=embedding_matrix)
                            bounds = self.hp.get_bounds()

                            acq_func = self.get_acq_func(train_x_embedded, self.train_y)

                            while True:
                                try:
                                    best_candidate, best_acq_value = optimize_acqf(
                                        acq_func, bounds=bounds_embedded, q=1, num_restarts=5, raw_samples=20,
                                    )
                                    break

                                except:
                                    pass"""

                            best_embedding_matrix = None
                            best_embedding_matrix_index = None
                            best_candidate = None
                            best_acq_value = None

                            for i, embedding_matrix in enumerate(self.matrix_manger):
                                train_x_embedded = torch.t(torch.mm(embedding_matrix, torch.t(self.train_x)))
                                bounds_embedded = self.hp.get_bounds(embedding_matrix=embedding_matrix)
                                bounds = self.hp.get_bounds()

                                acq_func = self.get_acq_func(train_x_embedded, self.train_y)

                                # get acquisition
                                while True:
                                    candidate, acq_value = optimize_acqf(
                                        acq_func, bounds=bounds_embedded, q=1, num_restarts=5, raw_samples=20,
                                    )

                                    if acq_value is not None:
                                        break

                                if best_acq_value is None or acq_value > best_acq_value:

                                    best_embedding_matrix = deepcopy(embedding_matrix)
                                    best_embedding_matrix_index = i
                                    best_candidate = candidate
                                    best_acq_value = acq_value

                                # update embedding matrix
                            self.matrix_manger.update_matrices_except_best_one(best_embedding_matrix_index)

                            new_train_x = torch.t(torch.mm(
                                torch.inverse(torch.mm(torch.t(best_embedding_matrix), best_embedding_matrix)),
                                torch.mm(torch.t(best_embedding_matrix), torch.t(best_candidate))
                            ))[0]

                            new_train_x[new_train_x < bounds[0]] = bounds[0][new_train_x < bounds[0]]
                            new_train_x[new_train_x > bounds[1]] = bounds[1][new_train_x > bounds[1]]

                            hp_fix = self.hp.get_const_hp_from_tensor(new_train_x)

                            # print(f"embedding: {embedding_matrix}")
                            # print(f"embedding_bounds: {bounds_embedded}")

                            if self.logger is not None:
                                self.logger.info(f"The candidate of next_x is {new_train_x}, acquisition is {best_acq_value}.")


                        trainer = self.TrainerClass(self.root,
                                                    str(self.hp.experiment.experiment_name),
                                                    iter_id,
                                                    hp_fix,
                                                    embedding_id=embedding_id,
                                                    logger=self.logger)

                        self.thread_manager.add_task(trainer, new_train_x)
                        iter_id += 1

                time.sleep(0.1)

        # calc contribution

        return

    def get_best_result(self):
        return self.best_x, self.best_y

    def get_contribution_rate(self):
        pass

    def get_acq_func(self, train_x, train_y):
        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        if self.acq_func_type == TYPE_ACQUISITION_FUNC_UCB:
            return UpperConfidenceBound(gp, beta=0.1)
        elif self.acq_func_type == TYPE_ACQUISITION_FUNC_EI:
            return ExpectedImprovement(gp, best_f=0.2)
        elif self.acq_func_type == TYPE_ACQUISITION_FUNC_PI:
            return ProbabilityOfImprovement(gp, best_f=0.2)
        elif self.acq_func_type == TYPE_ACQUISITION_FUNC_POSTEIORMEAN:
            return PosteriorMean(gp)

    def get_candidates(self):
        pass


class ThreadManager:
    """
    Class for Experiment using Basyan Optimization.

    Attributes:
        num_initial_iterations (int): the number of initial iterations before sampling based on baysian optimization.
        num_max_iterations (int): the number of iterations containing initial iterations.
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """

    # attributes

    def __init__(self, devices, num_iterations, logger=None):
        """関数の説明タイトル
        Initializer.

        Args:
            hp (:obj:`引数の型`, optional): The hyperparams of experiment that contain training parameter.
            trainer (:obj:`引数の型`, optional): The training instance.
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
        self.logger = logger
        self.device_dict = {device: False for device in self.devices}
        self.q_task_pending = Queue()
        self.q_task_finished = Queue()
        self.list_task_running = []
        self.num_iterations = num_iterations
        self.num_task_finished = 0

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

                    if self.q_task_pending.qsize() > 0:
                        _trainer, new_train_x = self.q_task_pending.get()
                        task_thread = self.__launch_task_thread(_trainer, new_train_x, task.device)
                        task_thread.setDaemon(True)

                        self.list_task_running.append(task_thread)
                        self.list_task_running[-1].start()
                    else:
                        self.device_dict[task.device] = False

                    self.num_task_finished += 1
                    self.list_task_running.pop(i)
                    break

            time.sleep(0.1)

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
                task_thread.setDaemon(True)
                self.device_dict[key] = True

                self.list_task_running.append(task_thread)
                self.list_task_running[-1].start()
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
        if self.q_task_pending.qsize() > 0:
            return False

        if self.num_task_finished < self.num_iterations:
            return False

        for key, value in self.device_dict.items():
            # check if value is None
            if value:
                return False

        return True


