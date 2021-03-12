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

import itertools

from pathlib import Path
from copy import deepcopy

# third party modules
import numpy as np
import yaml

import torch

# original modules
from utils.const import *
from utils.utils import *


def scaling(value, max, min, scale_type=TYPE_SCALE_LINEAR):
    if isinstance(max, list):
        max_ = max[0]
    else:
        max_ = max
    if isinstance(min, list):
        min_ = min[0]
    else:
        min_ = min

    if isinstance(value, list):
        return [scaling(value_, max_, min_, scale_type=scale_type) for value_ in value]
    else:
        if scale_type == TYPE_SCALE_LINEAR:
            return (value - min_) / (max_ - min_)
        elif scale_type == TYPE_SCALE_LOG:
            return (np.log(value) - np.log(min_)) / (np.log(max_) - np.log(min_))
        elif scale_type == TYPE_SCALE_EXP:
            return (np.exp(value) - np.exp(min_)) / (np.exp(max_) - np.exp(min_))
        elif scale_type.startswith("x"):
            try:
                power = float(scale_type.replace("x"))
            except:
                raise Exception("nanka kaku!")
            return (value ** power - min_ ** power) / (max_ ** power - min_ ** power)
        else:
            raise Exception("nanka kaku!")


def unscaling(value, max, min, scale_type=TYPE_SCALE_LINEAR):
    if isinstance(max, list):
        max_ = max[0]
    else:
        max_ = max
    if isinstance(min, list):
        min_ = min[0]
    else:
        min_ = min

    if isinstance(value, list):
        return [unscaling(value_, max_, min_, scale_type=scale_type) for value_ in value]
    else:
        if scale_type == TYPE_SCALE_LINEAR:
            return value * (max_ - min_) + min_
        elif scale_type == TYPE_SCALE_LOG:
            return np.exp(value * (np.log(max_) - np.log(min_)) + np.log(min_))
        elif scale_type == TYPE_SCALE_EXP:
            return np.log(value * (np.exp(max_) - np.exp(min_)) + np.exp(min_))
        elif scale_type.startswith("x"):
            try:
                power = float(scale_type.replace("x"))
            except:
                raise Exception("nanka kaku!")
            return (value * (max_ ** power - min_ ** power) + min_ ** power) ** (1. / power)


class HP:
    def __init__(self, name, params=None, file_path=None, parent=None, logger=None):
        self.child_hp = []
        self.child_var = []
        self.child_const = []
        self.num_param_dim = None
        self._has_discrete = False
        self._is_variable = False
        self.index = 0

        self.parent = parent
        self.logger = logger

        self.name = name

        if params is None and file_path is None:
            raise Exception(f"nanka kaku!")

        # if file path is specified. read the yaml file
        if file_path is not None:
            with open(file_path, "r") as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)

        for key, value in params.items():
            if isinstance(value, dict):
                # check if having necessary keys
                if "is_variable" in value:
                    if value["is_variable"] in ["True", "true", True]:
                        self.set_var(key, value)
                        self.child_var.append(key)
                        self._is_variable = True
                        if getattr(self, key).get_type() in [TYPE_VALUE_BOOL, TYPE_VALUE_INT, TYPE_VALUE_CATEGORICAL]:
                            self._has_discrete = True

                    elif value["is_variable"] in ["False", "false", False]:
                        self.set_const(key, value)
                        self.child_const.append(key)

                    else:
                        raise Exception(f"nanka kaku!")

                else:
                    setattr(self, key, HP(key, params=value, parent=self.name, logger=self.logger))
                    self.child_hp.append(getattr(self, key))
                    append_list = [f"{key}.{var_name}" for var_name in getattr(self, key).child_var]

                    self.child_var += append_list
                    self.child_const += [f"{key}.{const_name}" for const_name in getattr(self, key).child_const]
                    self._has_discrete = getattr(self, key).has_discrete() or self._has_discrete
                    self._is_variable = getattr(self, key).is_variable()

            elif isinstance(value, str) or \
                    isinstance(value, list) or \
                    isinstance(value, int) or \
                    isinstance(value, float):
                self.set_const(key, value)
                self.child_const.append(key)

            else:
                raise Exception(f"nanka kaku!")

        self.num_param_dim = sum([self.get_var(var_name).get_dim() for var_name in self.child_var])

    def set_const(self, name, param):
        if isinstance(param, dict):
            if not ("is_variable" in param and (param["is_variable"] == "False").all()):
                raise Exception(f"nanka kaku!")

            type = ALL_TYPE_VALUE[str(param["type"])] if "type" in param else None
            default_value = (param["default_value"]) if "default_value" in param else None
            is_list = isinstance(default_value, list)

            setattr(self, name, Constant(name, default_value, float(param), type=type, parent=self, is_list=is_list))

        elif isinstance(param, str) or isinstance(param, int) or isinstance(param, float):
            try:
                # first, if value can be casted to int, treat as int
                setattr(self, name, Constant(name, int(param), type=TYPE_VALUE_INT, parent=self))

            except ValueError:
                try:
                    # then, if value can be casted to float, treat as float
                    setattr(self, name, Constant(name, float(param), type=TYPE_VALUE_FLOAT, parent=self))
                except ValueError:
                    # then, if value is "True" or "False" , treat as bool
                    if param in ["True", "False"]:
                        setattr(self,
                                name,
                                Constant(
                                    name,
                                    bool(param),
                                    type=TYPE_VALUE_BOOL,
                                    parent=self
                                )
                                )
                    else:
                        # then, treat as str
                        setattr(self, name, Constant(name, param, type=TYPE_VALUE_STR, parent=self))

        elif isinstance(param, list):
            param_list = []
            element_type = None
            for element in param:
                try:
                    # first, if value can be casted to int, treat as int
                    element = int(element)
                    param_list.append(element)
                    if element_type is None:
                        element_type = TYPE_VALUE_INT
                    elif element_type != TYPE_VALUE_INT:
                        raise Exception(f"nanka kaku!")
                except ValueError:
                    try:
                        # then, if value can be casted to float, treat as float
                        element = float(element)
                        param_list.append(element)
                        if element_type is None:
                            element_type = TYPE_VALUE_FLOAT
                        elif element_type != TYPE_VALUE_FLOAT:
                            raise Exception(f"nanka kaku!")
                    except ValueError:
                        # then, if value is "True" or "False" , treat as bool
                        if param == "True" or param == "False":
                            element = bool(element)
                            param_list.append(element)
                            if element_type is None:
                                element_type = TYPE_VALUE_BOOL
                            elif element_type != TYPE_VALUE_BOOL:
                                raise Exception(f"nanka kaku!")

                        else:
                            # then, treat as str
                            param_list.append(element)
                            if element_type is None:
                                element_type = TYPE_VALUE_STR
                            elif element_type != TYPE_VALUE_STR:
                                raise Exception(f"nanka kaku!")

            setattr(self,
                    name,
                    Constant(name,
                             param,
                             type=element_type,
                             is_list=True,
                             parent=self)
                    )

        else:
            raise Exception(f"nanka kaku!")

    def set_var(self, name, param):
        # check
        if not ("is_variable" in param and (param["is_variable"] in ["True", "true", True])):
            raise Exception(f"nanka kaku!")

        type = ALL_TYPE_VALUE[str(param["type"])] if "type" in param else None
        scale = ALL_TYPE_SCALE[str(param["scale"])] if "scale" in param else None
        candidates = param["candidates"] if "candidates" in param else None
        max_value = param["max_value"] if "max_value" in param else None
        min_value = param["min_value"] if "min_value" in param else None

        setattr(self,
                name,
                Variable(name,
                         type=type,
                         scale=scale,
                         candidates=candidates,
                         parent=self,
                         max_value=max_value,
                         min_value=min_value)
                )

    def get(self, level):
        if "." in level:
            attr_name_first = level.split(".")[0]
            attr_name_else = ".".join(level.split(".")[1:])
            return (getattr(self, attr_name_first)).get(attr_name_else)
        else:
            return getattr(self, level).get_value()

    def is_variable(self):
        return self._is_variable

    def get_const_hp_from_tensor(self, tensor):
        # check tensor dim
        if tensor.size(0) != self.num_param_dim:
            raise Exception(f"nanka kaku!")

        header_index = 0

        hp_const = deepcopy(self)

        for i, var_name in enumerate(self.child_var):
            var_dim = self.get_var(var_name).get_dim()
            const = hp_const.get_var(var_name).get_const_from_tensor(tensor[header_index:header_index + var_dim])
            hp_const.var2const(var_name, const)
            header_index += var_dim

        return hp_const

    def get_var_dict_from_tensor(self, tensor):
        output = {}
        if isinstance(tensor, str):
            return output
        if isinstance(tensor, list):
            tensor = torch.tensor(tensor)

        # check tensor dim
        if tensor.size(0) != self.num_param_dim:
            raise Exception(f"nanka kaku!")

        header_index = 0

        for i, var_name in enumerate(self.child_var):
            var_dim = self.get_var(var_name).get_dim()
            const = self.get_var(var_name).get_const_from_tensor(tensor[header_index:header_index + var_dim])
            header_index += var_dim

            output[var_name] = const.get_value()

        return output

    def generate_random_sample(self, param, with_hp=False):
        tensor_list = []

        for i, var_name in enumerate(self.child_var):
            tensor_list.append(self.get_var(var_name).get_random_tensor().float().view(-1))

        param_tensor = torch.cat(tensor_list)

        if with_hp:
            return param_tensor, self.get_const_hp_from_tensor(param_tensor)
        else:
            return param_tensor

    def var2const(self, var_name, const):
        if var_name not in self.child_var:
            raise Exception(f"nanka kaku!")

        self.child_var.remove(var_name)
        self.child_const.append(var_name)

        top_attr_name = var_name.split(".")[0]
        if not hasattr(self, top_attr_name):
            raise Exception(f"nanka kaku!")

        else:
            target_attr = getattr(self, top_attr_name)
            if isinstance(target_attr, Variable):
                # target_attr = const
                delattr(self, top_attr_name)
                setattr(self, top_attr_name, const)
            elif isinstance(target_attr, HP):
                return target_attr.var2const(var_name[len(top_attr_name) + 1:], const)
            else:
                raise Exception(f"nanka kaku!")

    def get_var(self, var_name):
        if var_name not in self.child_var:
            raise Exception(f"nanka kaku!")

        top_attr_name = var_name.split(".")[0]
        if not hasattr(self, top_attr_name):
            print(var_name, top_attr_name)
            print(self.child_var)
            print(self.name)
            raise Exception(f"nanka kaku!")

        else:
            target_attr = getattr(self, top_attr_name)
            if isinstance(target_attr, Variable):
                return target_attr
            elif isinstance(target_attr, HP):
                return target_attr.get_var(var_name[len(top_attr_name) + 1:])
            else:
                raise Exception(f"nanka kaku!")

    def get_const(self, const_name):
        if const_name not in self.child_const:
            raise Exception(f"nanka kaku!")

        top_attr_name = const_name.split(".")[0]
        if not hasattr(self, top_attr_name):
            raise Exception(f"nanka kaku!")

        else:
            target_attr = getattr(self, top_attr_name)
            if isinstance(target_attr, Constant):
                return target_attr
            elif isinstance(target_attr, HP):
                return target_attr.get_var(const_name[len(top_attr_name) + 1:])
            else:
                print(type(target_attr))
                raise Exception(f"nanka kaku!")

    def discrete_patterns(self):
        header_index = 0
        candidates_list = []
        index_list = []
        output = []
        # list up all discrete param
        for i, var_name in enumerate(self.child_var):
            var = self.get_var(var_name)
            if var.type in (TYPE_VALUE_FLOAT, ):
                # Nothing to do
                pass
            elif var.type in (TYPE_VALUE_CATEGORICAL, ):
                candidates_list.append(var.candidates_onehot)
                index_list += list(range(header_index, header_index + var.get_dim()))
            elif var.type in (TYPE_VALUE_INT, TYPE_VALUE_BOOL):
                candidates_list.append(var.candidates_scaled)
                index_list += list(range(header_index, header_index + var.get_dim()))
            else:
                raise Exception(f"nanka kaku!")

            header_index += var.get_dim()

        if len(candidates_list) == 0:
            return {}
        else:
            all_patterns = itertools.product(*candidates_list)

            for pattern in all_patterns:
                fixed_features = {}
                i = 0
                for j, param in enumerate(pattern):
                    if isinstance(param, (int, bool, float)) == 1:
                        fixed_features[index_list[i]] = float(param)
                        i += 1
                    else:
                        for k in range(len(param)):
                            fixed_features[index_list[i]] = float(param[0][k])
                            i += 1

                output.append(fixed_features)

            return output

    def get_name(self):
        return self.name

    def get_bounds(self, embedding_matrix=None):
        if embedding_matrix is not None:
            min_bounds = []
            max_bounds = []
            for i in range(embedding_matrix.size(0)):
                min_value = 0
                max_value = 0
                header_index = 0
                # for j in range(embedding_matrix.size(1)):
                #     min_value += min(float(embedding_matrix[i, j]), 0.)
                #     max_value += max(float(embedding_matrix[i, j]), 0.)
                for j, var_name in enumerate(self.child_var):
                    var = self.get_var(var_name)
                    if var.type in (TYPE_VALUE_FLOAT,):
                        min_value += min(float(embedding_matrix[i, header_index]), 0.)
                        max_value += max(float(embedding_matrix[i, header_index]), 0.)
                    elif var.type in (TYPE_VALUE_CATEGORICAL, TYPE_VALUE_INT, TYPE_VALUE_BOOL):
                        min_value += min(torch.min(embedding_matrix[i, header_index: header_index + var.get_dim()]), 0.)
                        max_value += max(torch.max(embedding_matrix[i, header_index: header_index + var.get_dim()]), 0.)
                    else:
                        raise Exception(f"nanka kaku!")

                    header_index += var.get_dim()

                min_bounds.append(min_value)
                max_bounds.append(max_value)

            min_bounds = torch.tensor(min_bounds)
            max_bounds = torch.tensor(max_bounds)

        else:
            min_bounds = torch.tensor([0.]*self.num_param_dim)
            max_bounds = torch.tensor([1.]*self.num_param_dim)

        return torch.stack([min_bounds, max_bounds])

    def print_all(self, tab=0):
        print("\t" * tab + f"{self.name}:")

        tab += 1

        for var_name in self.child_var:
            if len(var_name.split(".")) == 1:
                print(
                    "\t" * tab + f"{self.get_var(var_name).get_name()} (variable) : {self.get_var(var_name).get_value()}")
            else:
                pass

        for const_name in self.child_const:
            if len(const_name.split(".")) == 1:
                print(
                    "\t" * tab + f"{self.get_const(const_name).get_name()} (const) : {self.get_const(const_name).get_value()}")
            else:
                pass

        for hp in self.child_hp:
            hp.print_all(tab=tab)

    def check_params(self, required_params):
        for required_param in required_params:
            if required_param not in self.child_var and required_param not in self.child_const:
                raise Exception(f"nanka kaku!")

    def has_discrete(self):
        return self._has_discrete

    def to_dict(self):
        output = {}
        for hp in self.child_hp:
            output[hp.name] = hp.to_dict()

        for const_name in self.child_const:
            const_name_split = const_name.split(".")
            if len(const_name_split) == 1:
                const_instance = self.get_const(const_name)
                output[const_name] = const_instance.to_dict()

        for var_name in self.child_var:
            var_name_split = var_name.split(".")
            if len(var_name_split) == 1:
                var_instance = self.get_var(var_name)
                output[var_name] = var_instance.to_dict()

        return output

    def get_dim(self):
        return self.num_param_dim

    def __len__(self):
        return len(self.child_hp) + len(self.child_var) + len(self.child_const)

    def __next__(self):
        if self.index == self.__len__():
            raise StopIteration

        elif self.index >= len(self.child_hp) + len(self.child_var):
            ret = self.child_const[self.index - len(self.child_hp) - len(self.child_var)]
        elif self.index >= len(self.child_hp):
            ret = self.child_var[self.index - len(self.child_hp)]
        else:
            ret = self.child_hp[self.index]

        self.index += 1
        return ret


class Variable:
    """
    Base class for Training. This class will be inherited by other task-specified Trainer class such as SegTrainer, CLSTrainer, and CNTTrainer.

    Attributes:
        属性の名前 (属性の型): 属性の説明
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """

    def __init__(self, var_name, type=None, scale=None, candidates=None, parent=None, max_value=None, min_value=None):
        """関数の説明タイトル
        Initizer function.

        Args:
            引数の名前 (引数の型): 引数の説明
            引数の名前 (:obj:`引数の型`, optional): 引数の説明.
        Returns:
            戻り値の型: 戻り値の説明 (例 : True なら成功, False なら失敗.)
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

        self.name = var_name
        self.type = type
        self.scale = scale
        self.candidates = candidates

        self.parent = parent
        self.max_value = [max_value]
        self.min_value = [min_value]

        self.candidates_scaled = None
        self.candidates_onehot = None
        self.num_categories = None

        # check if arguments is appropriate and init it.
        self.check_and_init_args()

        self.dim = 1 if self.type not in [TYPE_VALUE_CATEGORICAL] else len(self.candidates)

        if self.candidates is not None:
            self.default_value = self.candidates[0]
        else:
            self.default_value = self.min_value

    def check_and_init_args(self):
        # check args
        if self.type in [TYPE_VALUE_INT] and (
                (self.max_value is None or self.min_value is None) and self.candidates is None):
            raise Exception("nanka kaku!")

        elif self.type == TYPE_VALUE_CATEGORICAL and self.candidates is None:
            raise Exception("nanka kaku!")

        elif self.type in [TYPE_VALUE_FLOAT] and (self.max_value is None or self.min_value is None):
            raise Exception("nanka kaku!")

        # if scale type is not specified, set linear.
        if self.scale is None:
            self.scale = TYPE_SCALE_LINEAR

        # bool
        if self.type == TYPE_VALUE_BOOL:
            self.max_value = [1.]
            self.min_value = [0.]
            self.candidates = [1., 0.]
            self.candidates_scaled = self.candidates

        elif self.type == TYPE_VALUE_CATEGORICAL:
            # generate one-hot vector
            self.num_categories = len(self.candidates)
            self.candidates_onehot = [np.eye(1, M=self.num_categories, k=i, dtype=np.int8) for i in
                                      range(self.num_categories)]
            self.candidates_scaled = self.candidates_onehot
            self.max_value = [1. for i in range(self.num_categories)]
            self.min_value = [0. for i in range(self.num_categories)]

        elif self.type == TYPE_VALUE_INT:
            if self.candidates is not None:
                self.max_value = [max(self.candidates)]
                self.min_value = [min(self.candidates)]
            else:
                if self.max_value is None or self.min_value is None:
                    raise Exception("nanka kaku!")
                else:
                    self.candidates = list(range(self.min_value[-1], self.max_value[-1] + 1))

            self.candidates_scaled = scaling(self.candidates,
                                             self.max_value[-1],
                                             self.min_value[-1],
                                             self.scale)

        elif self.type == TYPE_VALUE_FLOAT:
            if self.max_value is None or self.min_value is None:
                raise Exception("nanka kaku!")

        else:
            raise Exception("nanka kaku!")

        return True

    def get_random_tensor(self):
        # if candidates is not None, choose value from candidates.
        if self.candidates is not None:
            index = np.random.choice(list(range(len(self.candidates))))
            if self.type == TYPE_VALUE_BOOL:
                return torch.tensor([self.candidates[index]])
            elif self.type == TYPE_VALUE_INT:
                return scaling(torch.tensor([self.candidates[index]]), self.max_value, self.min_value, self.scale)
            elif self.type == TYPE_VALUE_CATEGORICAL:
                return torch.tensor(self.candidates_onehot[index])

        else:
            while True:
                # TO_DO
                value = (self.max_value[0] - self.min_value[0]) * np.random.rand() + self.min_value[0]
                if self.scale == TYPE_SCALE_EXP and value == 0:
                    continue
                else:
                    if self.type == TYPE_VALUE_BOOL:
                        return torch.tensor([value])
                    elif self.type == TYPE_VALUE_INT:
                        return scaling(torch.tensor([value]), self.max_value, self.min_value,
                                       self.scale)
                    elif self.type == TYPE_VALUE_CATEGORICAL:
                        return torch.tensor(value)
                    return torch.tensor([value])

    def get_const_from_tensor(self, tensor_value):
        if self.type == TYPE_VALUE_CATEGORICAL:
            for i in range(len(self.candidates)):
                if tensor_value[..., i] == 1:
                    return Constant(self.name, self.candidates[i], TYPE_VALUE_CATEGORICAL)
                else:
                    max_values, max_indices = torch.max(tensor_value, 0)
                    # print(f"hogehogehoge, {max_values}, {max_indices}")
                    return Constant(self.name, self.candidates[max_indices], TYPE_VALUE_CATEGORICAL)

        elif self.type == TYPE_VALUE_BOOL:
            return Constant(self.name, bool(tensor_value), TYPE_VALUE_BOOL)

        elif self.type == TYPE_VALUE_INT:
            return Constant(self.name,
                            int(unscaling(tensor_value, self.max_value, self.min_value, self.scale)),
                            TYPE_VALUE_INT)
        elif self.type == TYPE_VALUE_FLOAT:
            return Constant(self.name,
                            float(unscaling(tensor_value, self.max_value, self.min_value, self.scale)),
                            TYPE_VALUE_FLOAT)

        else:
            raise Exception("nanka kaku!")

    def to_dict(self):
        output = {"is_variable": True}
        if self.type is not None:
            output["type"] = inverse_lookup(ALL_TYPE_VALUE, self.type)
        if self.scale is not None:
            output["scale"] = inverse_lookup(ALL_TYPE_SCALE, self.scale)
        if self.candidates is not None:
            output["candidates"] = self.candidates
        if self.max_value is not None:
            output["max_value"] = self.max_value[0]
        if self.min_value is not None:
            output["min_value"] = self.min_value[0]
        if self.default_value is not None:
            output["default_value"] = self.default_value

        return output

    def get_max(self):
        return self.max_value

    def get_min(self):
        return self.min_value

    def get_bounds(self):
        return self.max_value, self.min_value

    def get_value(self):
        return self.default_value

    def keys(self, group=None):
        pass

    def values(self, group=None):
        pass

    def items(self, group=None):
        pass

    def get_name(self):
        return self.name

    def get_dim(self):
        return self.dim

    def get_type(self):
        return self.type

    def __len__(self):
        return 1


class Constant:
    """
    Base class for Training. This class will be inherited by other task-specified Trainer class such as SegTrainer, CLSTrainer, and CNTTrainer.

    Attributes:
        属性の名前 (属性の型): 属性の説明
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """

    def __init__(self, const_name, value, type, parent=None, is_list=False):
        """関数の説明タイトル
        Initizer function.

        Args:
            引数の名前 (引数の型): 引数の説明
            引数の名前 (:obj:`引数の型`, optional): 引数の説明.
        Returns:
            戻り値の型: 戻り値の説明 (例 : True なら成功, False なら失敗.)
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
        self.name = const_name
        self.value = value
        self.type = type
        self.is_list = is_list
        self.parent = parent

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def to_dict(self):
        output = {"is_variable": False}
        if self.type is not None:
            output["type"] = inverse_lookup(ALL_TYPE_VALUE, self.type)
        if self.value is not None:
            output["value"] = self.value

        return output

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __len__(self):
        if self.is_list:
            return len(self.value)
        else:
            return 1

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)
