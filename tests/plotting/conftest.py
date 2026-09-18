import matplotlib.pyplot as plt
import numpy as np
import pytest

ECDYSIS_INDEX = np.array([0.0, 25.0, 50.0, 75.0, 99.0])
N_FRAMES = 100


@pytest.fixture(autouse=True)
def shown_figures(monkeypatch):
    """
    Mimic the notebook inline backend, where ``plt.show()`` renders and then
    closes the current figure, so the next plot starts on a fresh one.
    """
    shown = []

    def show(*args, **kwargs):
        figure = plt.gcf()
        shown.append(figure)
        plt.close(figure)

    monkeypatch.setattr(plt, "show", show)
    return shown


def _make_condition(condition_id, n_worms, length_scale, rng):
    """
    A condition dict shaped like the output of ``build_plotting_struct``.

    Worms grow exponentially over 100 frames (10 min apart) with ecdysis every
    25 frames. Length follows volume ** (1 / 3) times ``length_scale``, so the
    length/volume proportion model is a straight line in log-log space.
    """
    time = np.tile(np.arange(N_FRAMES, dtype=float), (n_worms, 1))
    experiment_time = time * 600
    ecdysis_index = np.tile(ECDYSIS_INDEX, (n_worms, 1))
    durations = np.diff(ecdysis_index, axis=1)
    worm_sizes = rng.lognormal(0, 0.1, (n_worms, 1))
    volume = 1e4 * worm_sizes * np.exp(0.03 * time)
    length = length_scale * volume ** (1 / 3)
    molt_frames = ECDYSIS_INDEX.astype(int)
    return {
        "condition_id": condition_id,
        "description": f"condition {condition_id}",
        "food": "OP50",
        "point": np.arange(n_worms)[:, np.newaxis] + 10 * condition_id,
        "time": time,
        "experiment_time": experiment_time,
        "experiment_time_hours": experiment_time / 3600,
        "ecdysis_index": ecdysis_index,
        "ecdysis_time_step": ecdysis_index,
        "ecdysis_experiment_time": ecdysis_index * 600,
        "ecdysis_experiment_time_hours": ecdysis_index / 6,
        "larval_stage_durations_time_step": durations,
        "larval_stage_durations_experiment_time_hours": durations / 6,
        "body_seg_volume": volume,
        "body_seg_length": length,
        "body_seg_qc": np.full((n_worms, N_FRAMES), "worm", dtype=object),
        "body_seg_volume_at_ecdysis": volume[:, molt_frames],
        "body_seg_length_at_ecdysis": length[:, molt_frames],
    }


@pytest.fixture
def conditions_struct():
    """Control (id 0) and a condition whose worms are 10% longer (id 1)."""
    rng = np.random.default_rng(0)
    return [
        _make_condition(0, n_worms=30, length_scale=1.0, rng=rng),
        _make_condition(1, n_worms=30, length_scale=1.1, rng=rng),
    ]
