"""Incremental (stage-wise) training of unfolded solvers.

Requires the optional ``train`` extra (``ray[tune]``); the core ``ucmf`` package
stays Ray-free. :func:`train_model` serves Ray Tune and plain single runs alike:
every validation point is reported to Ray (a no-op outside a Ray session) and
collected in the returned :class:`~ucmf.train.result.TrainingResult`.
"""

from functools import partial
from time import perf_counter

from ..optim.neural_augmentation import NeuralAugmentation
from ..scaling import normalize_operator
from .objectives import build_loss, nmse_loss
from .result import TrainingResult
from .utils import (
    batch_operator,
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
    validation_batches,
)

import torch


def train_model(config, *, y_train, y_val, q_train=None, q_val=None, index_train=None, index_val=None):
    """Train an unfolded solver incrementally over stages ``t = 1..config.stages``.

    Stage ``t`` runs ``config.steps_per_stage`` Adam steps at solver depth ``t`` and
    restores the stage's best state by validation loss (as in FA2026). The model
    solves with the column-normalized operator ``D = A U^-1``; supervised losses map its
    output back to source strengths, see :mod:`ucmf.scaling`. With
    ``config.data_scaling='equireg'``, measurements and targets are divided per sample
    by ``lambda_max`` of the normalized operator. Validation runs in fixed batches of
    ``config.batch_size`` (incomplete batch dropped, see :func:`validation_batches`). Every
    ``config.val_every`` steps and at the end of each stage, one metrics dict is
    reported to Ray and appended to the history. With ``q_val``, it also holds
    ``val_nmse``, the validation NMSE of the final iterate for any loss; checkpoint
    selection still uses ``val_loss``. ``train_seconds`` times the steps since the previous
    point (including bookkeeping such as stage changes), ``val_seconds`` the validation
    (including ``val_nmse``). The existing syncs ``loss.item()`` and the validation's
    ``.item()`` bound both; compilation shows up in the first ``train_seconds`` (training
    graph) and the first ``val_seconds`` (no-grad graph).

    A step with a non-finite loss or gradient norm is rejected (as in FA2026): the model
    returns to the stage's best state so far and Adam restarts with fresh moments, the
    learning rate unchanged. ``rejected_steps`` in the history counts them since the
    previous point. The check syncs the GPU once per step.

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
        operator; required for 'nmse'. ``q_val`` adds ``val_nmse`` to the history.
    index_train, index_val : Tensor, optional
        Bank entry of every sample, shape ``(N,)`` / ``(M,)``, required exactly if the
        configured operator is a dictionary bank (per problem, ``K`` entries). Training
        batches select their dictionaries by ``config.batch_sampling``.

    Returns
    -------
    TrainingResult
        History, best validation loss of the last stage, and the trained state.
    """
    operator, column_norms = normalize_operator(build_operator(config))
    y_train, q_train, index_train = prepare_data(config, operator, y_train, q_train, index_train)
    y_val, q_val, index_val = prepare_data(config, operator, y_val, q_val, index_val)
    val_batches = validation_batches(config, y_val, q_val, index_val)
    model = build_model(config)
    objective = build_loss(config)
    val_nmse = partial(nmse_loss, w=0)  # final iterate with the full operator, for any loss
    solve = build_solve(config, model)
    if isinstance(model, NeuralAugmentation):
        problems = None if index_train is None else index_train[: config.batch_size]
        model.init_normalizers(y_train[: config.batch_size], batch_operator(operator, column_norms, problems)[0])
    batches = iterate_batches(config, y_train.shape[0], index_train)
    best_state = clone_state(model)
    history = []
    rejected_steps = 0
    last = perf_counter()

    for stage in range(1, config.stages + 1):
        optimizer = build_optimizer(config, model)
        copy_state(best_state, model)  # fallback if no validation point of this stage is finite
        stage_best_val_loss = float('inf')
        for step in range(1, config.steps_per_stage + 1):
            optimizer.zero_grad()
            samples, problems = next(batches)
            q_batch = q_train[samples] if config.supervised else None
            batch_op, batch_norms = batch_operator(operator, column_norms, problems)
            loss = objective(solve, y_train[samples], q_batch, stage, batch_op, batch_norms)
            loss.backward()
            parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm([parameter.grad for parameter in parameters])
            if torch.isfinite(loss) & torch.isfinite(grad_norm):  # the per-step sync
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grads_with_norm_(parameters, config.grad_clip_norm, grad_norm)
                optimizer.step()
            else:  # reject: back to the stage's best state, Adam without the corrupt moments
                model.load_state_dict(best_state)
                optimizer = build_optimizer(config, model)
                rejected_steps += 1

            if step % config.val_every != 0 and step != config.steps_per_stage:
                continue
            train_loss = loss.item()  # syncs the GPU, so train_seconds covers the computed steps
            train_seconds, last = perf_counter() - last, perf_counter()
            val_loss = evaluate(objective, solve, val_batches, operator, column_norms, stage)
            metrics = {
                'stage': stage,
                'step': (stage - 1) * config.steps_per_stage + step,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'rejected_steps': rejected_steps,
            }
            rejected_steps = 0
            if q_val is not None:
                metrics['val_nmse'] = evaluate(val_nmse, solve, val_batches, operator, column_norms, stage)
            metrics['train_seconds'], metrics['val_seconds'] = train_seconds, perf_counter() - last
            if val_loss < stage_best_val_loss:
                stage_best_val_loss = val_loss
                copy_state(best_state, model)
            metrics['stage_best_val_loss'] = stage_best_val_loss
            history.append(metrics)
            ray_report(metrics)
            last = perf_counter()  # bookkeeping counts toward the next train_seconds
        model.load_state_dict(best_state)

    return TrainingResult(
        config=config,
        history=history,
        best_val_loss=stage_best_val_loss,
        state_dict={key: value.cpu() for key, value in best_state.items()},
    )
