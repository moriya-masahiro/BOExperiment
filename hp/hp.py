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


def scaling(value, max, min, scale_type=TYPE_SCALE_LINEAR):
    if scale_type == TYPE_SCALE_LINEAR:
        return (value - min) / (max - min)
    elif scale_type == TYPE_SCALE_LOG:
        return (np.log(value) - np.log(min)) / (np.log(max) - np.log(min))
    elif scale_type == TYPE_SCALE_EXP:
        return (np.exp(value) - np.exp(min)) / (np.exp(max) - np.exp(min))
    elif scale_type.startswith("x"):
        try:
            power = float(scale_type.replace("x"))
        except:
            raise Exception("nanka kaku!")
        return (value ** power - min ** power) / (max ** power - min ** power)
    else:
        raise Exception("nanka kaku!")


def unscaling(value, max, min, scale_type=TYPE_SCALE_LINEAR):
    if scale_type == TYPE_SCALE_LINEAR:
        return value * (max - min) + min
    elif scale_type == TYPE_SCALE_LOG:
        return np.exp(value * (np.log(max) - np.log(min)) + np.log(min))
    elif scale_type == TYPE_SCALE_EXP:
        return np.log(value * (np.exp(max) - np.exp(min)) + np.exp(min))
    elif scale_type.startswith("x"):
        try:
            power = float(scale_type.replace("x"))
        except:
            raise Exception("nanka kaku!")
        return (value * (max ** power - min ** power) + min ** power) ** (1. / power)


class HP:
    def __init__(self, name, params=None, file_path=None, parent=None):
        self.parent = parent
        self.child_hp = []
        self.child_var = []
        self.child_const = []
        self.name = name
        self.num_param_dim = None

        is_var = None

        self.index = 0

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
                    elif value["is_variable"] in ["False", "false", False]:
                        self.set_const(key, value)
                        self.child_const.append(key)
                    else:
                        raise Exception(f"nanka kaku!")

                else:
                    setattr(self, key, HP(key, params=value))
                    self.child_hp.append(getattr(self, key))
                    self.child_var += [f"{key}.{var_name}" for var_name in getattr(self, key).child_var]
                    self.child_const += [f"{key}.{const_name}" for const_name in getattr(self, key).child_const]

            elif isinstance(value, str) or \
                    isinstance(value, list) or \
                    isinstance(value, int) or \
                    isinstance(value, float):
                self.set_const(key, value)

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

            except:
                try:
                    # then, if value can be casted to float, treat as float
                    setattr(self, name, Constant(name, float(param), type=TYPE_VALUE_FLOAT, parent=self))
                except:
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
                    param_list.append(int(element))
                    if element_type is None:
                        element_type = TYPE_VALUE_INT
                    elif element_type != TYPE_VALUE_INT:
                        raise Exception(f"nanka kaku!")
                except:
                    try:
                        # then, if value can be casted to float, treat as float
                        param_list.append(float(param))
                        if element_type is None:
                            element_type = TYPE_VALUE_FLOAT
                        elif element_type != TYPE_VALUE_FLOAT:
                            raise Exception(f"nanka kaku!")
                    except:
                        # then, if value is "True" or "False" , treat as bool

                        param_list.append(float(param))
                        if element_type is None:
                            element_type = TYPE_VALUE_FLOAT
                        elif element_type != TYPE_VALUE_FLOAT:
                            raise Exception(f"nanka kaku!")

                        if param == "True" or param == "False":
                            param_list.append(bool(param))
                            if element_type is None:
                                element_type = TYPE_VALUE_BOOL
                            elif element_type != TYPE_VALUE_BOOL:
                                raise Exception(f"nanka kaku!")

                        else:
                            # then, treat as str
                            param_list.append(param)
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
        if self.is_variable:
            return True
        # if child variable is not empty, return True
        elif len(self.child_var):
            self.is_variable = True
            return True
        else:
            # check all child hp
            for child_hp in self.child_hp:
                if child_hp.is_variable():
                    self.is_variable = True
                    return True
            else:
                self.is_variable = False
                return False

    def get_const_hp_from_tensor(self, tensor):
        # check tensor dim
        if tensor.size(0) != self.num_param_dim:
            raise Exception(f"nanka kaku!")

        header_index = 0

        hp_const = deepcopy(self)

        for i, var_name in enumerate(self.child_var):
            var_dim = self.get_var(var_name).get_dim()
            const = hp_const.get_var(var_name).get_const_from_tensor(tensor[header_index:header_index+var_dim])
            hp_const.var2const(var_name, const)
            header_index += var_dim

        # hp_const.child_const = deepcopy(hp_const.child_var)
        # hp_const.child_var = []

        return hp_const

    def generate_random_sample(self, param, with_hp=False):
        tensor_list = []

        for i, var_name in enumerate(self.child_var):
            tensor_list.append(self.get_var(var_name).get_random_tensor())

        param_tensor = torch.cat(tensor_list)

        if with_hp:
            return param_tensor, self.get_const_hp_from_tensor(param_tensor)
        else:
            return param_tensor

    def var2const(self, var_name, const):
        if var_name not in self.child_var:
            print(var_name)
            print(self.child_var)
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
                return target_attr.var2const(var_name[len(top_attr_name)+1:], const)
            else:
                raise Exception(f"nanka kaku!")

    def get_var(self, var_name):
        if var_name not in self.child_var:
            print(var_name)
            print(self.child_var)
            raise Exception(f"nanka kaku!")

        top_attr_name = var_name.split(".")[0]
        if not hasattr(self, top_attr_name):
            raise Exception(f"nanka kaku!")

        else:
            target_attr = getattr(self, top_attr_name)
            if isinstance(target_attr, Variable):
                return target_attr
            elif isinstance(target_attr, HP):
                return target_attr.get_var(var_name[len(top_attr_name)+1:])
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
                return target_attr.get_var(const_name[len(top_attr_name)+1:])
            else:
                print(type(target_attr))
                raise Exception(f"nanka kaku!")

    def discrete_patterns(self):
        header_index = 0
        candidates_list = []
        index_list = []
        output = []
        # list up all discrete param
        for i, var in enumerate(self.child_var):

            if (var.type == "float").all():
                # Nothing to do.
                pass
            elif (var.type == "categorical").all():
                candidates_list.append(var.candidates_onehot)
                index_list.append(header_index)
            elif var.type in ["int", "bool"]:
                candidates_list.append(var.candidates)
                index_list.append(header_index)
            else:
                Exception(f"nanka kaku!")

            header_index += var.get_dim()

        num_discrete_param = len(candidates_list)

        if len(candidates_list) == 0:
            return {}
        else:
            all_patterns = itertools.product(*candidates_list)

            for pattern in all_patterns:
                pattern_dict = {}
                for i, param in enumerate(pattern):
                    if len(param) == 1:
                        pattern_dict[index_list[i]] = float(param)
                    else:
                        for j in range(len(param)):
                            pattern_dict[index_list[i] + j] = float(param[j])

                output.append(pattern_dict)

            return output

    def get_name(self):
        return self.name

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

    def __iter__(self):
        return self

    def get_bounds(self):
        return torch.stack([torch.zeros(self.num_param_dim), torch.ones(self.num_param_dim)])

    def print_all(self, tab=0):
        print("\t" * tab + f"{self.name}:")

        tab += 1

        for var_name in self.child_var:
            if len(var_name.split(".")) == 1:
                print("\t" * tab + f"{self.get_var(var_name).get_name()} (variable) : {self.get_var(var_name).get_value()}")
            else:
                pass

        for const_name in self.child_const:
            if len(const_name.split(".")) == 1:
                print("\t" * tab + f"{self.get_const(const_name).get_name()} (const) : {self.get_const(const_name).get_value()}")
            else:
                pass

        for hp in self.child_hp:
            hp.print_all(tab=tab)


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
        self.max_value = max_value
        self.min_value = min_value

        # check if arguments is appropriate and init it.
        self.check_and_init_args()

        self.dim = 1 if self.type not in [TYPE_VALUE_CATEGORICAL] else len(self.candidates)

        if self.candidates is not None:
            self.default_value = self.candidates[0]
        else:
            self.default_value = self.min_value

    def check_and_init_args(self):
        # check args
        if self.type in [TYPE_VALUE_INT] and ((self.max_value is None or self.min_value is None) or self.candidates is None):
            raise Exception("nanka kaku!")

        elif self.type == TYPE_VALUE_CATEGORICAL and self.candidates is None:
            raise Exception("nanka kaku!")

        elif self.type in [TYPE_VALUE_FLOAT] and (self.max_value is None or self.min_value is None):
            raise Exception("nanka kaku!")

        # if scale type is not supecified, set linear
        if self.scale is None:
            self.scale = TYPE_SCALE_LINEAR

        # bool
        if self.type == TYPE_VALUE_BOOL:
            self.max_value = 1
            self.min_value = 0
            self.candidates = [1, 0]

        elif self.type == TYPE_VALUE_CATEGORICAL:
            # generate one-hot vector
            self.num_categories = len(self.candidates)
            self.candidates_onehot = [np.eye(1, M=self.num_categories, k=i, dtype=np.int8) for i in range(self.num_categories)]
            self.max_value = 1
            self.min_value = 0

        elif self.type == TYPE_VALUE_INT:
            if self.candidates is not None:
                self.max_value = max(self.candidates)
                self.min_value = min(self.candidates)
            else:
                if self.max_value is None or self.min_value is None:
                    raise Exception("nanka kaku!")
                else:
                    self.candidates = list(range(self.min_value, self.max_value+1))

        elif self.type == TYPE_VALUE_FLOAT:
            if self.max_value is None or self.min_value is None:
                raise Exception("nanka kaku!")

        else:
            raise Exception("nanka kaku!")

        return True

    def get_random_tensor(self):
        if self.candidates is not None:
            index = np.random.choice(list(range(len(self.candidates))))
            if self.type == TYPE_VALUE_BOOL:
                return torch.tensor([self.candidates[index]])
            elif self.type == TYPE_VALUE_INT:
                return scaling(torch.tensor([self.candidates[index]]), self.max, self.min, self.scale)
            elif self.type == TYPE_VALUE_CATEGORICAL:
                return torch.tensor(self.candidates_onehot[index])

        else:
            while True:
                value = np.random.rand()
                if self.scale == TYPE_SCALE_EXP and value == 0:
                    continue
                else:
                    return torch.tensor([value])

    def get_const_from_tensor(self, tensor_value):
        if self.type == TYPE_VALUE_CATEGORICAL:
            for i in range(len(self.candidates)):
                if tensor_value[..., i] == 1:
                    return Constant(self.name, i, TYPE_VALUE_CATEGORICAL)

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
