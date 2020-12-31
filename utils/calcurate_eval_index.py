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

# third party modules
import numpy as np

import torch
import torch.nn.functional as F

# original modules
from torch import Tensor

from ..utils.const import *


def calc_ap(precisions, recalls, batch=True):
    # check dim
    if not torch.all(torch.eq(precisions.size(), recalls.size())):
        raise Exception("nanka kaku!")

    with torch.no_grad():
        if batch:
            # sort recall list and precision list
            sorted_recalls, idx = torch.sort(recalls)
            sorted_precisions = precisions[idx]

            # padding
            sorted_recalls_padded = F.pad(sorted_recalls, (1, 0), mode="const", value=0)
            sorted_recalls_padded: Tensor = F.pad(sorted_recalls_padded, (0, 1), mode="const", value=1)

            sorted_precisions_padded = F.pad(sorted_precisions, mode="replicate")

            # calculate ap
            ap = (sorted_precisions_padded[..., :-1] + sorted_precisions_padded[..., 1:]) / 2. \
                 * (sorted_recalls_padded[..., 1:] - sorted_recalls_padded[..., :-1])

        else:
            # sort recall list and precision list
            sorted_recalls, idx = torch.sort(recalls)
            sorted_precisions = precisions[idx]

            # padding
            sorted_recalls_padded = F.pad(sorted_recalls, (1, 0), mode="const", value=0)
            sorted_recalls_padded: Tensor = F.pad(sorted_recalls_padded, (0, 1), mode="const", value=1)

            sorted_precisions_padded = F.pad(sorted_precisions, mode="replicate")

            # calculate ap
            ap = (sorted_precisions_padded[..., :-1] + sorted_precisions_padded[..., 1:]) / 2. \
                     * (sorted_recalls_padded[..., 1:] - sorted_recalls_padded[..., :-1])

    return ap


def calc_map(precisions, recalls, batch=True):
    # check dim
    if not torch.all(torch.eq(precisions.size(), recalls.size())):
        raise Exception("nanka kaku!")

    with torch.no_grad():
        if batch:
            # sort recall list and precision list
            sorted_recalls, idx = torch.sort(recalls)
            sorted_precisions = precisions[idx]

            # padding
            sorted_recalls_padded = F.pad(sorted_recalls, (1, 0), mode="const", value=0)
            sorted_recalls_padded: Tensor = F.pad(sorted_recalls_padded, (0, 1), mode="const", value=1)

            sorted_precisions_padded = F.pad(sorted_precisions, mode="replicate")

            # calculate ap
            ap = (sorted_precisions_padded[..., :-1] + sorted_precisions_padded[..., 1:]) / 2. \
                 * (sorted_recalls_padded[..., 1:] - sorted_recalls_padded[..., :-1])

        else:
            # sort recall list and precision list
            sorted_recalls, idx = torch.sort(recalls)
            sorted_precisions = precisions[idx]

            # padding
            sorted_recalls_padded = F.pad(sorted_recalls, (1, 0), mode="const", value=0)
            sorted_recalls_padded: Tensor = F.pad(sorted_recalls_padded, (0, 1), mode="const", value=1)

            sorted_precisions_padded = F.pad(sorted_precisions, mode="replicate")

            # calculate ap
            ap = (sorted_precisions_padded[..., :-1] + sorted_precisions_padded[..., 1:]) / 2. \
                 * (sorted_recalls_padded[..., 1:] - sorted_recalls_padded[..., :-1])

    return ap


def calc_iou(gt, pred, is_batch=True, ignore_class=255):
    # check dim
    if not torch.all(torch.eq(gt.size(), pred.size())):
        raise Exception("nanka kaku!")

    with torch.no_grad():
        if is_batch:
            # calc inter area and union area
            ignore = torch.eq(gt, ignore_class)
            inter = torch.count_nonezero(torch.eq(gt, pred) * ignore, dim=1)
            union = torch.count_nonezero(gt * ignore, dim=1) + torch.count_nonezero(pred * ignore, dim=1) - inter

            # if union = 0, iou = 0
            iou = ~torch.eq(union, 0) * (inter + 1e-16) / (union + 1e-16)

        else:
            # calc inter area and union area
            inter = torch.count_nonezero(torch.eq(gt, pred), dim=0)
            union = torch.count_nonezero(gt, dim=0) + torch.count_nonezero(gt, dim=0) - inter

            # if union = 0, iou = 0
            iou = ~torch.eq(union, 0) * (inter + 1e-16) / (union + 1e-16)

    return iou


def calc_miou(gt, pred, is_batch=True, num_classes=None, ignore_class=255, classwise=False):
    # check dim
    if not torch.all(torch.eq(gt.size(), pred.size())):
        raise Exception("nanka kaku!")

    # check class id
    if num_classes is None:
        num_classes = torch.max(torch.cat([gt, pred]))

    ious = []
    with torch.no_grad():
        for i in range(num_classes):
            gt_one_class = torch.eq(gt, i)
            pred_one_class = torch.eq(pred, i)
            ious.append(calc_iou(gt_one_class, pred_one_class, is_batch=is_batch, ignore_class=ignore_class))

    return torch.mean(torch.cat(ious)), torch.cat(ious) if classwise else torch.mean(torch.cat(ious))


def calc_precision(gt, pred, is_batch=True, ignore_class=255):
    # check dim
    if not torch.all(torch.eq(gt.size(), pred.size())):
        raise Exception("nanka kaku!")

    with torch.no_grad():
        if is_batch:
            # calc true positive and positive
            ignore = torch.eq(gt, ignore_class)
            tp = torch.count_nonezero(torch.eq(gt, pred) * ignore, dim=1)
            p = torch.count_nonezero(pred * ignore, dim=1)

            # if positive = 0, precision = 0
            precision = ~torch.eq(p, 0) * (tp + 1e-16) / (p + 1e-16)


        else:
            # calc true positive and positive
            ignore = torch.eq(gt, ignore_class)
            tp = torch.count_nonezero(torch.eq(gt, pred) * ignore)
            p = torch.count_nonezero(pred * ignore)

            # if positive = 0, precision = 0
            precision = ~torch.eq(p, 0) * (tp + 1e-16) / (p + 1e-16)

    return precision


def calc_mprecision(gt, pred, is_batch=True, num_classes=None, ignore_class=255, classwise=False):
    # check dim
    if not torch.all(torch.eq(gt.size(), pred.size())):
        raise Exception("nanka kaku!")

    # check class id
    if num_classes is None:
        num_classes = torch.max(torch.cat([gt, pred]))

    precisions = []
    with torch.no_grad():
        for i in range(num_classes):
            gt_one_class = torch.eq(gt, i)
            pred_one_class = torch.eq(pred, i)
            precisions.append(calc_precision(gt_one_class, pred_one_class, is_batch=is_batch, ignore_class=ignore_class))

    return torch.mean(torch.cat(precisions)), torch.cat(precisions) if classwise else torch.mean(torch.cat(precisions))


def calc_recall(gt, pred, is_batch=True, ignore_class=255):
    # check dim
    if not torch.all(torch.eq(gt.size(), pred.size())):
        raise Exception("nanka kaku!")

    with torch.no_grad():
        if is_batch:
            # calc true positive and positive of gt
            ignore = torch.eq(gt, ignore_class)
            tp = torch.count_nonezero(torch.eq(gt, pred) * ignore, dim=1)
            p_gt = torch.count_nonezero(gt * ignore, dim=1)

            # if positive of gt = 0, recall = 1
            recall = torch.eq(p_gt, 0) + ~torch.eq(p_gt, 0) * (tp + 1e-16) / (p_gt + 1e-16)


        else:
            # calc true positive and positive of gt
            ignore = torch.eq(gt, ignore_class)
            tp = torch.count_nonezero(torch.eq(gt, pred) * ignore)
            p_gt = torch.count_nonezero(gt * ignore)

            # if positive of gt = 0, recall = 1
            recall = torch.eq(p_gt, 0) + ~torch.eq(p_gt, 0) * (tp + 1e-16) / (p_gt + 1e-16)

    return recall


def calc_mrecall(gt, pred, is_batch=True, num_classes=None, ignore_class=255, classwise=False):
    # check dim
    if not torch.all(torch.eq(gt.size(), pred.size())):
        raise Exception("nanka kaku!")

    # check class id
    if num_classes is None:
        num_classes = torch.max(torch.cat([gt, pred]))

    recalls = []
    with torch.no_grad():
        for i in range(num_classes):
            gt_one_class = torch.eq(gt, i)
            pred_one_class = torch.eq(pred, i)
            recalls.append(calc_recall(gt_one_class, pred_one_class, is_batch=is_batch, ignore_class=ignore_class))

    return torch.mean(torch.cat(recalls)), torch.cat(recalls) if classwise else torch.mean(torch.cat(recalls))