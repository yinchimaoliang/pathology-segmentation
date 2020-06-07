import numpy as np

from pathseg.core.evals import build_eval


def test_iou():
    iou_cfg = dict(type='Iou', class_num=2)

    iou = build_eval(iou_cfg)

    pred = np.array([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4, 0.4]]],
                     [[[0.4, 0.4], [0.4, 0.4]], [[0.6, 0.6], [0.6, 0.6]]]])
    gt = np.array([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]],
                   [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]])

    result = iou.step(pred, gt)
    assert abs(result - 0.5) < 1e-3


def test_dsc():
    dsc_cfg = dict(type='Dsc', class_num=2)

    dsc = build_eval(dsc_cfg)

    pred = np.array([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4, 0.4]]],
                     [[[0.4, 0.4], [0.4, 0.4]], [[0.6, 0.6], [0.6, 0.6]]]])
    gt = np.array([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]],
                   [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]])

    result = dsc.step(pred, gt)
    assert abs(result - 2 / 3) < 1e-3
