"""Tests for ucmf.train incremental training; skipped without the ``train`` extra."""

import pytest

pytest.importorskip('ray')

from ucmf.train import TrainingConfig, TrainingResult, train_model  # noqa: E402
from ucmf.train.utils import batch_loss, iterate_batches  # noqa: E402

import torch  # noqa: E402

M, N_DIM, N_SAMPLES = 6, 10, 16
STAGES, STEPS = 2, 3
MODEL_KWARGS = {'num_iterations': 3, 'gamma_init': 0.3, 'theta_init': 0.01, 'num_measurements': M}


@pytest.fixture
def problem():
    torch.manual_seed(0)
    A = torch.randn(M, N_DIM, dtype=torch.float64)
    q = torch.rand(N_SAMPLES, N_DIM, dtype=torch.float64).clamp_min(0)
    return A, q @ A.T, q


def make_config(A, **overrides):
    config = {
        'model_name': 'AdaLFISTASW',
        'operator_name': 'MatrixMult',
        'model_kwargs': MODEL_KWARGS,
        'operator_kwargs': {'matrix': A},
        'stages': STAGES,
        'steps_per_stage': STEPS,
        'batch_size': 4,
        'learning_rate': 1e-2,
        'loss': 'nmse',
        'device': 'cpu',
        'precision': 'float64',
    }
    return TrainingConfig(**{**config, **overrides})


def run(problem, **overrides):
    A, y, q = problem
    return train_model(make_config(A, **overrides), y_train=y, y_val=y[:4], q_train=q, q_val=q[:4])


def test_history_matches_reported_metrics(problem):
    result = run(problem)
    assert isinstance(result, TrainingResult)
    assert [entry['step'] for entry in result.history] == list(range(1, STAGES * STEPS + 1))
    assert set(result.history[0]) == {'stage', 'step', 'train_loss', 'val_loss', 'stage_best_val_loss'}
    assert set(result.state_dict) == {'B', '_gamma', '_theta'}


def test_best_val_loss_belongs_to_last_stage(problem):
    result = run(problem)
    last_stage = [entry['val_loss'] for entry in result.history if entry['stage'] == STAGES]
    assert result.best_val_loss == min(last_stage) == result.history[-1]['stage_best_val_loss']


def test_batches_cover_each_epoch_without_replacement():
    config = make_config(torch.eye(M, dtype=torch.float64), batch_size=5)
    batches = iterate_batches(config, N_SAMPLES)
    epoch = torch.cat([next(batches) for _ in range(4)])  # 5 + 5 + 5 + 1
    assert sorted(epoch.tolist()) == list(range(N_SAMPLES))


@pytest.mark.parametrize('loss', ['mse', 'nmse'])
def test_supervised_loss_compares_source_strengths(loss):
    config = make_config(torch.eye(M, dtype=torch.float64), loss=loss)
    column_norms = torch.rand(N_DIM, dtype=torch.float64) + 0.5
    q = torch.rand(4, N_DIM, dtype=torch.float64)
    assert batch_loss(config, column_norms * q, None, q, None, column_norms).item() == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(('loss', 'per_column'), [('relative_residual', True), ('nmse', False)])
def test_training_is_invariant_to_scaling_of_A(problem, loss, per_column):
    """``A diag(s)`` with targets ``q / s`` normalizes to the same problem.

    The NMSE in ``q`` weights the columns by ``1 / s``; it is invariant only to a global scale.
    """
    A, y, q = problem
    s = torch.rand(N_DIM, dtype=torch.float64) + 0.5 if per_column else torch.tensor(3.0, dtype=torch.float64)
    scaled = train_model(make_config(A * s, loss=loss), y_train=y, y_val=y[:4], q_train=q / s, q_val=q[:4] / s)
    reference = run(problem, loss=loss)
    for entry, expected in zip(scaled.history, reference.history, strict=True):
        assert entry['train_loss'] == pytest.approx(expected['train_loss'])


def test_unsupervised_loss_needs_no_targets(problem):
    A, y, _ = problem
    result = train_model(make_config(A, loss='relative_residual'), y_train=y, y_val=y[:4])
    assert result.best_val_loss >= 0


def test_supervised_loss_requires_targets(problem):
    A, y, _ = problem
    with pytest.raises(ValueError, match='requires q_train and q_val'):
        train_model(make_config(A), y_train=y, y_val=y[:4])


def test_invalid_config_rejected(problem):
    A, _, _ = problem
    for overrides, match in [
        ({'model_name': 'Nope'}, 'model_name'),
        ({'model_name': 'FISTA'}, 'model_name'),
        ({'operator_name': 'Nope'}, 'operator_name'),
        ({'loss': 'nope'}, 'loss'),
        ({'stages': 0}, 'stages'),
        ({'batch_size': 0}, 'batch_size'),
        ({'precision': 'float16'}, 'precision'),
    ]:
        with pytest.raises(ValueError, match=match):
            make_config(A, **overrides)


def test_operator_must_match_config_precision(problem):
    with pytest.raises(ValueError, match='operator must have dtype'):
        run(problem, precision='float32')


def test_na_model_initializes_normalizers(problem):
    model_kwargs = {'gamma_init': 0.3, 'theta_init': 0.01, 'lstm_dim': 5, 'num_measurements': M}
    result = run(problem, model_name='NA_AdaLFISTASW', model_kwargs=model_kwargs)
    assert len(result.history) == STAGES * STEPS
