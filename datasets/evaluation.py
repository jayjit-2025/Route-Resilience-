"""Evaluation metrics for road segmentation and graph topology.

Computes IoU, Dice, Connectivity Ratio, and Topological Accuracy
by comparing predicted masks/graphs against ground truth.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Segmentation metrics ──────────────────────────────────────────────────────

def compute_iou(pred: np.ndarray, gt: np.ndarray, tolerance: int = 0) -> float:
    """Intersection over Union between predicted and ground-truth road masks.

    Args:
        pred: (H, W) binary predicted mask {0, 1}.
        gt:   (H, W) binary ground-truth mask {0, 1}.
        tolerance: Pixel tolerance buffer (0 = strict, 3-5 = relaxed IoU).

    Returns:
        IoU score in [0, 1].
    """
    if pred.shape != gt.shape:
        gt = cv2.resize(gt.astype(np.uint8), (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

    p = (pred > 0).astype(np.uint8)
    g = (gt   > 0).astype(np.uint8)

    if tolerance > 0:
        kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
        g = cv2.dilate(g, kernel)

    intersection = np.logical_and(p, g).sum()
    union        = np.logical_or(p, g).sum()
    return float(intersection / union) if union > 0 else 1.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice / F1 score between predicted and ground-truth road masks.

    Returns:
        Dice score in [0, 1].
    """
    if pred.shape != gt.shape:
        gt = cv2.resize(gt.astype(np.uint8), (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

    p = (pred > 0).astype(np.uint8)
    g = (gt   > 0).astype(np.uint8)

    intersection = np.logical_and(p, g).sum()
    denom        = p.sum() + g.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


def compute_relaxed_iou(pred: np.ndarray, gt: np.ndarray,
                        tolerance: int = 3) -> float:
    """Length-Complete / Relaxed IoU with pixel tolerance buffer.

    If the predicted road pixel falls within ``tolerance`` pixels of the
    ground-truth road it counts as a true positive.

    Args:
        tolerance: Buffer size in pixels (default 3).
    """
    return compute_iou(pred, gt, tolerance=tolerance)


# ── Graph / topology metrics ──────────────────────────────────────────────────

def compute_connectivity_ratio(graph) -> float:
    """Fraction of nodes in the largest connected component."""
    try:
        import networkx as nx
        if graph.number_of_nodes() == 0:
            return 0.0
        components = list(nx.connected_components(graph))
        largest    = max(len(c) for c in components)
        return largest / graph.number_of_nodes()
    except Exception as e:
        logger.warning("Connectivity ratio computation failed: %s", e)
        return 0.0


def compute_topological_accuracy(
    pred_graph,
    gt_graph,
    n_pairs: int = 100,
) -> float:
    """Average path length error between predicted and GT road graphs.

    Samples ``n_pairs`` random node pairs, computes shortest path in both
    graphs, and returns mean relative error. Lower = better.

    Args:
        pred_graph: Predicted NetworkX road graph.
        gt_graph:   Ground-truth NetworkX road graph (from OSM).
        n_pairs:    Number of random node pairs to sample.

    Returns:
        Mean relative path length error in [0, ∞). 0 = perfect.
    """
    try:
        import networkx as nx
        import random

        pred_nodes = list(pred_graph.nodes())
        gt_nodes   = list(gt_graph.nodes())

        if len(pred_nodes) < 2 or len(gt_nodes) < 2:
            return 1.0

        errors = []
        for _ in range(n_pairs):
            u, v = random.sample(pred_nodes, 2)
            try:
                pred_len = nx.shortest_path_length(pred_graph, u, v)
                # Map to nearest GT nodes by node index proximity
                gt_u = min(gt_nodes, key=lambda n: abs(n - u))
                gt_v = min(gt_nodes, key=lambda n: abs(n - v))
                gt_len = nx.shortest_path_length(gt_graph, gt_u, gt_v)
                if gt_len > 0:
                    errors.append(abs(pred_len - gt_len) / gt_len)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                errors.append(1.0)

        return float(np.mean(errors)) if errors else 1.0

    except Exception as e:
        logger.warning("Topological accuracy computation failed: %s", e)
        return 1.0


def evaluate_segmentation(
    pred_mask: np.ndarray,
    gt_mask:   np.ndarray,
    pred_graph=None,
    gt_graph=None,
    iou_tolerance: int = 3,
) -> dict:
    """Run all evaluation metrics and return results dict.

    Args:
        pred_mask:     Predicted binary road mask.
        gt_mask:       Ground-truth binary road mask.
        pred_graph:    Optional predicted NetworkX graph.
        gt_graph:      Optional ground-truth NetworkX graph.
        iou_tolerance: Pixel tolerance for relaxed IoU.

    Returns:
        Dict with keys: iou, dice, relaxed_iou, connectivity_ratio,
        topological_accuracy.
    """
    results = {
        "iou":                compute_iou(pred_mask, gt_mask, tolerance=0),
        "dice":               compute_dice(pred_mask, gt_mask),
        "relaxed_iou":        compute_relaxed_iou(pred_mask, gt_mask, iou_tolerance),
        "connectivity_ratio": compute_connectivity_ratio(pred_graph) if pred_graph else None,
        "topological_accuracy": (
            compute_topological_accuracy(pred_graph, gt_graph)
            if pred_graph and gt_graph else None
        ),
    }
    logger.info(
        "Evaluation — IoU: %.3f | Dice: %.3f | Relaxed IoU: %.3f",
        results["iou"], results["dice"], results["relaxed_iou"],
    )
    return results
