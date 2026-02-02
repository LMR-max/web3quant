from __future__ import annotations

from typing import Iterator, Tuple


def walk_forward_splits(
    n_samples: int,
    min_train_size: int,
    test_size: int,
    step_size: int,
    train_window: int | None = None,
) -> Iterator[Tuple[slice, slice]]:
    start_train = 0
    start_test = min_train_size

    while start_test + test_size <= n_samples:
        if train_window:
            start_train = max(start_test - train_window, 0)
        train_slice = slice(start_train, start_test)
        test_slice = slice(start_test, start_test + test_size)
        yield train_slice, test_slice
        start_test += step_size
