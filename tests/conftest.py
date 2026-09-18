import cv2
import matplotlib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

WORM_RADIUS = 10


def pytest_configure(config):
    # plotting functions call plt.show(), which must be a no-op under test
    matplotlib.use("Agg")


def _draw_curved_worm(shape: tuple[int, int], y_offset: int = 0) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x = np.arange(40, 281)
    y = 60 + y_offset + 25 * np.sin((x - 40) / 240 * np.pi)
    points = np.stack([x, y.astype(np.int32)], axis=1).astype(np.int32)
    cv2.polylines(mask, [points], False, 1, thickness=2 * WORM_RADIUS + 1)
    return mask


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def curved_worm_mask() -> np.ndarray:
    """Sine-shaped worm, 21 px thick with round caps, in a 120x320 uint8 mask."""
    return _draw_curved_worm((120, 320))


@pytest.fixture
def curved_worm_expected_length() -> float:
    """Arc length of the drawn midline plus the two round caps."""
    x = np.arange(40, 281)
    y = 60 + 25 * np.sin((x - 40) / 240 * np.pi)
    return float(np.sum(np.hypot(np.diff(x), np.diff(y)))) + 2 * WORM_RADIUS


@pytest.fixture
def curved_worm_image(curved_worm_mask, rng) -> np.ndarray:
    """Bright blurred worm on a dim noisy background, as a uint16 image."""
    signal = gaussian_filter(curved_worm_mask.astype(float), 2) * 3000
    noise = rng.normal(0, 20, curved_worm_mask.shape)
    return (signal + 200 + noise).astype(np.uint16)


@pytest.fixture
def curved_worm_stack() -> np.ndarray:
    """Three planes of the curved worm, drifting 3 px down per plane."""
    return np.stack([_draw_curved_worm((120, 320), y_offset=3 * i) for i in range(3)])


@pytest.fixture
def rectangle_mask() -> np.ndarray:
    """An already-straight worm: 20 px wide, 100 px long."""
    mask = np.zeros((40, 120), dtype=np.uint8)
    mask[10:30, 10:110] = 1
    return mask


@pytest.fixture
def molt_centers() -> tuple[int, int, int, int]:
    return (120, 240, 360, 480)


@pytest.fixture
def molting_volume(molt_centers) -> tuple[np.ndarray, np.ndarray]:
    """
    A 600-frame volume curve growing exponentially with four 30-frame lethargus
    plateaus centered on ``molt_centers``, preceded by 20 egg frames.
    """
    n_frames, n_egg_frames = 600, 20
    rate = np.full(n_frames, 0.012)
    for center in molt_centers:
        rate[center - 15 : center + 15] = 0.0
    volume = np.exp(np.log(3e4) + np.cumsum(rate))
    volume[:n_egg_frames] = 1e3
    worm_types = np.array(["egg"] * n_egg_frames + ["worm"] * (n_frames - n_egg_frames))
    return volume, worm_types
