"""Incremental (stage-wise) training of unfolded solvers.

Requires the optional ``train`` extra (``ray[tune]``); the core ``ucmf`` package
stays Ray-free. :func:`train_model` serves Ray Tune and plain single runs alike:
every validation point is reported to Ray (a no-op outside a Ray session) and
collected in the returned :class:`~ucmf.train.result.TrainingResult`.
"""

from ..optim.neural_augmentation import NeuralAugmentation
from ..scaling import normalize_operator
from .objectives import build_loss
from .result import TrainingResult
from .utils import (
    build_model,
    build_operator,
    build_optimizer,
    build_solve,
    clone_state,
    copy_state,
    evaluate,
    iterate_batches,
    prepare_data,
    ray_report,
)

import torch


def train_model(config, *, y_train, y_val, q_train=None, q_val=None):
    """Train an unfolded solver incrementally over stages ``t = 1..config.stages``.

    Stage ``t`` runs ``config.steps_per_stage`` Adam steps at solver depth ``t`` and
    restores the stage's best state by validation loss (as in FA2026). The model
    solves with the column-normalized operator ``D = A U^-1``; supervised losses map its
    output back to source strengths, see :mod:`ucmf.scaling`. With
    ``config.data_scaling='equireg'``, measurements and targets are divided per sample
    by ``lambda_max`` of the normalized operator. Every
    ``config.val_every`` steps and at the end of each stage, one metrics dict is
    reported to Ray and appended to the history.

    Parameters
    ----------
    config : TrainingConfig
        Training settings, including the operator to build. For Ray Tune, convert the
        dict passed to the trainable with ``TrainingConfig(**config)``.
    y_train : Tensor
        Training measurements of shape ``(N, m)``.
    y_val : Tensor
        Validation measurements of shape ``(M, m)`` for per-stage checkpoint selection.
    q_train, q_val : Tensor, optional
        Source strengths ``q`` of shape ``(N, n)`` / ``(M, n)`` for the unnormalized
        operator; required for 'nmse'.

    Returns
    -------
    TrainingResult
        History, best validation loss of the last stage, and the trained state.
    """
    operator, column_norms = normalize_operator(build_operator(config))
    y_train, y_val, q_train, q_val = prepare_data(config, operator, y_train, y_val, q_train, q_val)
    model = build_model(config)
    objective = build_loss(config)
    solve = build_solve(config, model)
    if isinstance(model, NeuralAugmentation):
        model.init_normalizers(y_train[: config.batch_size], operator)
    batches = iterate_batches(config, y_train.shape[0])
    best_state = clone_state(model)
    history = []

    for stage in range(1, config.stages + 1):
        optimizer = build_optimizer(config, model)
        copy_state(best_state, model)  # fallback if no validation point of this stage is finite
        stage_best_val_loss = float('inf')
        for step in range(1, config.steps_per_stage + 1):
            optimizer.zero_grad()
            indices = next(batches)
            y_batch = y_train[indices]
            q_batch = q_train[indices] if config.supervised else None
            loss = objective(solve, y_batch, q_batch, stage, operator, column_norms)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            if step % config.val_every != 0 and step != config.steps_per_stage:
                continue
            val_loss = evaluate(objective, solve, y_val, q_val, operator, column_norms, stage)
            if val_loss < stage_best_val_loss:
                stage_best_val_loss = val_loss
                copy_state(best_state, model)
            metrics = {
                'stage': stage,
                'step': (stage - 1) * config.steps_per_stage + step,
                'train_loss': loss.item(),
                'val_loss': val_loss,
                'stage_best_val_loss': stage_best_val_loss,
            }
            history.append(metrics)
            ray_report(metrics)
        model.load_state_dict(best_state)

    return TrainingResult(
        config=config,
        history=history,
        best_val_loss=stage_best_val_loss,
        state_dict={key: value.cpu() for key, value in best_state.items()},
    )
