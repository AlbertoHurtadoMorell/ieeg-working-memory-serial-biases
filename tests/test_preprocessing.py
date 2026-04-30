"""Tests for ieeg_wm.preprocessing — padding and remove_small_segments."""
import numpy as np
import pytest

from ieeg_wm.preprocessing import padding, remove_small_segments


class TestPadding:
    def test_basic_padding_extends_both_sides(self):
        # Cluster at indices [3, 4].
        # Left padding (padding_value=2): data[1:3] → indices 1, 2 set to 1.
        # Right padding: data[4:6] → indices 4, 5 set to 1 (4 was already 1).
        data = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=float)
        result = padding(data.copy(), padding_value=2)
        assert result[1] == 1
        assert result[2] == 1
        assert result[5] == 1
        assert result[6] == 0  # outside the padding window

    def test_padding_does_not_overflow_left_boundary(self):
        data = np.array([1, 1, 0, 0, 0, 0], dtype=float)
        result = padding(data.copy(), padding_value=3)
        assert result[0] == 1

    def test_padding_does_not_overflow_right_boundary(self):
        data = np.array([0, 0, 0, 0, 1, 1], dtype=float)
        result = padding(data.copy(), padding_value=3)
        assert result[-1] == 1

    def test_no_artifacts_returns_zeros(self):
        data = np.zeros(10, dtype=float)
        result = padding(data.copy(), padding_value=2)
        assert np.all(result == 0)

    def test_all_artifact_unchanged(self):
        data = np.ones(10, dtype=float)
        result = padding(data.copy(), padding_value=2)
        assert np.all(result == 1)

    def test_zero_padding_value_leaves_data_unchanged(self):
        data = np.array([0, 1, 0, 1, 0], dtype=float)
        result = padding(data.copy(), padding_value=0)
        np.testing.assert_array_equal(result, data)


class TestRemoveSmallSegments:
    def test_small_gap_is_filled(self):
        data = np.array([1, 1, 0, 1, 1], dtype=float)
        result = remove_small_segments(data.copy(), min_seg_length=3)
        assert result[2] == 1

    def test_long_gap_is_preserved(self):
        data = np.array([1, 0, 0, 0, 1], dtype=float)
        result = remove_small_segments(data.copy(), min_seg_length=3)
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] == 0

    def test_no_artifacts_all_clean_preserved(self):
        data = np.zeros(10, dtype=float)
        result = remove_small_segments(data.copy(), min_seg_length=2)
        assert np.all(result == 0)

    def test_all_artifacts_unchanged(self):
        data = np.ones(10, dtype=float)
        result = remove_small_segments(data.copy(), min_seg_length=2)
        assert np.all(result == 1)

    def test_multiple_small_gaps_all_filled(self):
        data = np.array([1, 0, 1, 0, 1], dtype=float)
        result = remove_small_segments(data.copy(), min_seg_length=2)
        assert np.all(result == 1)
