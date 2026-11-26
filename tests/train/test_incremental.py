"""Tests for ucmf.train incremental training; skipped without the ``train`` extra."""

import pytest

pytest.importorskip('ray')

from ucmf.train import TrainingResult, train_model  # noqa: E402
from ucmf.train.utils import iterate_batches  # noqa: E402

from configs import training_config  # noqa: E402

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
    settings = {'operator_name': 'MatrixMult', 'operator_kwargs': {'matrix': A}, 'model_kwargs': MODEL_KWARGS}
    return training_config(
        **{**settings, 'stages': STAGES, 'steps_per_stage': STEPS, 'learning_rate': 1e-2, **overrides}
    )


def run(problem, **overrides):
    A, y, q = problem
    return train_model(make_config(A, **overrides), y_train=y, y_val=y[:4], q_train=q, q_val=q[:4])


def test_history_matches_reported_metrics(problem):
    result = run(problem)
    assert isinstance(result, TrainingResult)
    assert [entry['step'] for entry in result.history] == list(range(1, STAGES * STEPS + 1))
    assert set(result.history[0]) == {'stage', 'step', 'train_loss', 'val_loss', 'stage_best_val_loss', 'val_nmse'}
    assert set(result.state_dict) == {'B', '_gamma', '_theta'}


def test_best_val_loss_belongs_to_last_stage(problem):
    result = run(problem)
    last_stage = [entry['val_loss'] for entry in result.history if entry['stage'] == STAGES]
    assert result.best_val_loss == min(last_stage) == result.history[-1]['stage_best_val_loss']


def test_batches_are_full_and_without_replacement_per_epoch():
    config = make_config(torch.eye(M, dtype=torch.float64), batch_size=5)
    batches = iterate_batches(config, N_SAMPLES)
    epoch = [next(batches)[0] for _ in range(3)]  # 16 // 5 full batches, one index dropped
    assert all(batch.shape == (5,) for batch in epoch)
    assert len(set(torch.cat(epoch).tolist())) == 15


def test_too_few_samples_for_one_batch_rejected(problem):
    with pytest.raises(ValueError, match='fewer than batch_size=32'):
        run(problem, batch_size=32)


def test_compiled_training_matches_eager(problem):
    torch._dynamo.reset()  # fresh compile cache: the recompile limit would silently fall back to eager
    compiled, eager = run(problem, compile=True).history, run(problem).history
    for entry, expected in zip(compiled, eager, strict=True):
        assert entry['train_loss'] == pytest.approx(expected['train_loss'], rel=1e-9)
        assert entry['val_loss'] == pytest.approx(expected['val_loss'], rel=1e-9)


def test_training_is_invariant_to_a_global_scale_of_A(problem):
    """``s A`` with targets ``q / s`` normalizes to the same problem."""
    A, y, q = problem
    scaled = train_model(make_config(3.0 * A), y_train=y, y_val=y[:4], q_train=q / 3.0, q_val=q[:4] / 3.0)
    for entry, expected in zip(scaled.history, run(problem).history, strict=True):
        assert entry['train_loss'] == pytest.approx(expected['train_loss'])


def test_equireg_is_invariant_to_per_sample_data_scaling(problem):
    """Equiregularization removes the per-sample scales before the solver sees the data."""
    A, y, q = problem
    c = torch.rand(N_SAMPLES, 1, dtype=torch.float64) * 10 + 0.1
    config = make_config(A)
    scaled = train_model(config, y_train=c * y, y_val=c[:4] * y[:4], q_train=c * q, q_val=c[:4] * q[:4])
    reference = run(problem)
    for entry, expected in zip(scaled.history, reference.history, strict=True):
        assert entry['train_loss'] == pytest.approx(expected['train_loss'])


def test_equireg_requires_nonnegative_model(problem):
    A, _, _ = problem
    model_kwargs = {**MODEL_KWARGS, 'nonnegative': False}
    with pytest.raises(ValueError, match='requires a nonnegative model'):
        make_config(A, model_kwargs=model_kwargs)
    make_config(A, model_kwargs=model_kwargs, data_scaling='none')


def test_weighted_iterates_change_the_loss(problem):
    # Stage 1 has one iterate, so both runs agree until the first step of stage 2.
    weighted, final = run(problem, w=0.5).history[STEPS], run(problem).history[STEPS]
    assert weighted['train_loss'] > final['train_loss']  # the earlier iterate adds a positive term


def test_val_nmse_is_recorded_for_any_loss_exactly_with_q_val(problem):
    A, y, q = problem
    config = make_config(A, loss='lasso', loss_kwargs={'lam': 0.1})
    with_q = train_model(config, y_train=y, y_val=y[:4], q_val=q[:4]).history
    without_q = train_model(config, y_train=y, y_val=y[:4]).history
    assert all('val_nmse' in entry for entry in with_q)
    assert not any('val_nmse' in entry for entry in without_q)


def test_val_nmse_equals_val_loss_for_unweighted_nmse(problem):
    for entry in run(problem).history:
        assert entry['val_nmse'] == pytest.approx(entry['val_loss'])


def test_supervised_loss_requires_targets(problem):
    A, y, _ = problem
    with pytest.raises(ValueError, match='requires source-strength targets'):
        train_model(make_config(A), y_train=y, y_val=y[:4])


def test_invalid_config_rejected(problem):
    A, _, _ = problem
    for overrides, match in [
        ({'model_name': 'Nope'}, 'model_name'),
        ({'model_name': 'FISTA'}, 'model_name'),
        ({'operator_name': 'Nope'}, 'operator_name'),
        ({'loss': 'nope'}, 'loss'),
        ({'w': 1.5}, 'w must lie'),
        ({'data_scaling': 'nope'}, 'data_scaling'),
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
