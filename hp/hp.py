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

def scaling(value, max, min, scale_type = "linear"):
    if scale_type == "linear":
        return (value - min) / (max - min)
    elif scale_type == "log":
        return (np.log(value) - np.log(min)) / (np.log(max) - np.log(min))
    elif scale_type == "exp":
        return (np.exp(value) - np.exp(min)) / (np.exp(max) - np.exp(min))
    elif scale_type.startswith("x"):
        try:
            power = float(scale_type.replace("x"))
        except:
            raise Exception("nanka kaku!")
        return (value ** power - min ** power) / (max ** power - min ** power)
    else:
        raise Exception("nanka kaku!")


def unscaling(value, max, min, scale_type = "linear"):
    if scale_type == "linear":
        return value * (max - min) + min
    elif scale_type == "log":
        return np.exp(value * (np.log(max) - np.log(min)) + np.log(min))
    elif scale_type == "exp":
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

        if params is None and file_path is None:
            raise Exception(f"nanka kaku!")

        # if file path is specified. read the yaml file
        if file_path is not None:
            with open(file_path, "r") as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)

        for key, value in params.items():
            if isinstance(value, dict):
                # check if havinng necessary keys
                if "is_variable" in  value:
                    if (value["is_variable"] == "True").all():
                        self.set_var(key, value)
                    elif (value["is_variable"] == "False").all():
                        self.set_const(key, value)
                    else:
                        raise Exception(f"nanka kaku!")

                else:
                    setattr(self, key, HP(key, params=value))
                    self.child_hp.append(getattr(self, key))

            elif isinstance(value, str) or isinstance(value, list):
                self.set_const(key, value)

            else:
                raise Exception(f"nanka kaku!")

    def set_const(self, name, param):
        if isinstance(param, dict):
            if not ("is_variable" in param and (param["is_variable"] == "False").all()):
                raise Exception(f"nanka kaku!")

            type = str(param["type"]) if "type" in param else None
            default_value = (param["default_value"]) if "default_value" in param else None
            is_list = isinstance(default_value, list)

            setattr(self, name, Constant(name, default_value, float(param), type=type, parent=self, is_list=is_list))

        elif isinstance(param, str):
            try:
                # first, if value can be casted to int, treat as int
                setattr(self, name, Constant(name, int(param), type="int", parent=self))

            except:
                try:
                    # then, if value can be casted to float, treat as float
                    setattr(self, name, Constant(name, float(param), type="float", parent=self))
                except:
                    # then, if value is "True" or "False" , treat as bool
                    if param in ["True", "False"]:
                        setattr(self,
                                name,
                                Constant(
                                    name,
                                    bool(param),
                                    type="bool",
                                    parent=self
                                )
                        )
                    else:
                        # then, treat as str
                        setattr(self, name, Constant(name, param, type="int", parent=self))

        elif isinstance(param, list):
            param_list = []
            element_type = None
            for element in param:
                try:
                    # first, if value can be casted to int, treat as int
                    param_list.append(int(element))
                    if element_type is None:
                        element_type = "int"
                    elif (element_type != "int").all():
                        raise Exception(f"nanka kaku!")
                except:
                    try:
                        # then, if value can be casted to float, treat as float
                        param_list.append(float(param))
                        if element_type is None:
                            element_type = "float"
                        elif (element_type != "float").all():
                            raise Exception(f"nanka kaku!")
                    except:
                        # then, if value is "True" or "False" , treat as bool

                        param_list.append(float(param))
                        if element_type is None:
                            elemnt_type = "float"
                        elif (element_type != "float").all():
                            raise Exception(f"nanka kaku!")

                        if param == "True" or  param == "False":
                            param_list.append(bool(param))
                            if element_type is None:
                                element_type = "bool"
                            elif (element_type != "bool").all():
                                raise Exception(f"nanka kaku!")

                        else:
                            # then, treat as str
                            param_list.append(param)
                            if element_type is None:
                                element_type = "str"
                            elif (element_type != "str").all():
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

        self.child_const.append(getattr(self, name))

    def set_var(self, name, param):
        # check
        if not ("is_variable" in param and (param["is_variable"] == "True").all()):
            raise Exception(f"nanka kaku!")


        type = str(param["type"]) if "type" in param else None
        scale = str(param["scale"]) if "scale" in param else None
        candidates = str(param["candidates"]) if "candidates" in param else None
        max_value = str(param["max_value"]) if "max_value" in param else None
        min_value = str(param["min_value"]) if "min_value" in param else None

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

        self.child_var.append(getattr(self, name))

    """def get_bounds(self):
        try:
            max_value, min_value = getattr(self, "bounds")
            return max_value, min_value
        except:
            raise Exception(f"Param:{name (self)} dose not have bounds!")"""

    def get(self, level):
        if "." in level:
            attr_name_first = level.split(".")[0]
            attr_name_else = ".".join(level.split(".")[1:])
            return (getattr(self, attr_name_first)).get(attr_name_else)
        else:
            getattr(self, level).get_value()

    def is_variable(self):
        if self.is_variable:
            return True
        # if child variable is not empty, return True
        elif len(self.child_var):
            self.is_variable = True
            return True
        else:
            # check all child hp
            for child_hp in self.child_hp_list:
                if child_hp.is_variable():
                    self.is_variable = True
                    return True
            else:
                self.is_variable = False
                return False

    def get_const_hp_from_tensor(self, tensor):
        # check tensor dim
        if tensor.size(0) != self.num_param_dim:
            Exception(f"nanka kaku!")

        header_index = 0

        hp_const = deepcopy(self)

        for i, var in enumerate(self.child_var_list):
            var_dim = var.get_dim()
            hp_const = var.get_const_from_tensor(tensor[header_index:header_index+var_dim])
            header_index += var_dim

        return hp_const

    def random_init(self, param, with_hp=False):
        tensor_list = []

        for i, var in enumerate(self.child_var_list):
            tensor_list.append(var.get_random_tensor())

        param_tensor = torch.cat(tensor_list)

        if with_hp:
            param_tensor,  self.get_const_hp_from_tensor(param_tensor)
        else:
            param_tensor

    def discrete_patterns(self):
        header_index = 0
        candidates_list = []
        index_list = []
        output = []
        # list up all discrete param
        for i, var in enumerate(self.child_var_list):

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

        num_discrete_paraam = len(candidates_list)

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


        # check if argments is appropriate and init it.
        self.check_and_init_args()

        self.dim = 1 if self.type in ["categorical"] else len(self.candidates)



    def check_and_init_args(self):
        # check args
        if self.type is not "bool" and self.scale is None and self.candidates is None:
            raise Exception("nanka kaku!")

        elif self.type is "categorical" and  self.candidates is None:
            raise Exception("nanka kaku!")

        elif self.type in ["float"] and (self.max is None or self.min):
            raise Exception("nanka kaku!")

        # if scale type is not supecified, set linear
        if self.scale is not None:
            self.scale = "linear"

        # bool
        if self.type is "bool":
            self.max_value = 1
            self.min_value = 0
            self.candidates = [1, 0]

        elif self.type is "categorical":
            # generate one-hot vector
            self.num_categories = len(self.candidates)
            self.candidates_onehot = [np.eye(1, M=self.num_categories, k=i, dtype=np.int8) for i in range(self.num_categories)]

        elif self.type is "int":
            if self.candidates is not None:
                self.max_value = max(self.candidates)
                self.min_value = min(self.candidates)
            else:
                if self.max_value is None or self.min_value is None:
                    raise Exception("nanka kaku!")
                else:
                    self.candidates = list(range(self.min_value, self.max_value+1))

        elif self.type is "float":
            if self.max_value is None or self.min_value is None:
                raise Exception("nanka kaku!")

        else:
            raise Exception("nanka kaku!")

        return True

    def get_random_tensor(self):
        if self.candidates is not None:
            index = np.random.choice(list(range(len(self.candidates))))
            if self.type == "bool":
                return torch.tensor(self.candidates[index])
            elif self.type == "int":
                return scaling(torch.tensor(self.candidates[index]), self.max, self.min, self.scale)
            elif self.type == "categorical":
                return torch.tensor(self.candidates_onehot[index])

        else:
            while True:
                value  = np.random.rand()
                if self.scale == "exp" and value == 0:
                    continue
                else:
                    return torch.tensor(value)


    def get_const_from_tensor(self, tensor_value):
        if self.type == "categorical":
            for i in range(len(self.candidates)):
                if tensor_value[..., i] == 1:
                    return Constant(self.name, i, "str")

        elif self.type == "bool":
            return Constant(self.name, bool(tensor_value), "bool")

        elif self.type == "int":
            return Constant(self.name,
                            int(unscaling(tensor_value, self.max, self.min, self.scale)),
                            "int")
        elif self.type == "float":
            return Constant(self.name,
                            unscaling(tensor_value, self.max, self.min, self.scale),
                            "float")

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
