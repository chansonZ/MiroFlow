# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the AgentPool (web_app/core/agent_pool.py).

All tests use mocked agents so that no real LLM or tool infrastructure is
needed.
"""

import queue
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(pool_size=3, max_overflow=-1, **kwargs):
    """Create an AgentPool with mocked build internals."""
    # Import here so that the module under test can be imported cleanly
    from web_app.core.agent_pool import AgentPool  # noqa: PLC0415

    pool = AgentPool(
        config_path="config/test.yaml",
        project_root=Path("/fake/root"),
        pool_size=pool_size,
        max_overflow=max_overflow,
        **kwargs,
    )
    return pool


def _mock_build(pool, num_agents=None):
    """Patch _build_one_agent to return dummy agents."""
    call_count = [0]
    if num_agents is None:
        num_agents = pool._pool_size

    def _fake_build(self):  # noqa: ANN001
        call_count[0] += 1
        agent = MagicMock(name=f"mock_agent_{call_count[0]}")
        return agent, 0.01

    return patch.object(type(pool), "_build_one_agent", _fake_build)


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_fills_pool_to_pool_size(self):
        pool = _make_pool(pool_size=3)
        with _mock_build(pool):
            built = pool.warmup()
        assert built == 3
        assert pool.available == 3

    def test_partial_warmup_on_build_failure(self):
        pool = _make_pool(pool_size=3)
        call_count = [0]

        def _flaky_build(self):  # noqa: ANN001
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("simulated build failure")
            agent = MagicMock(name=f"agent_{call_count[0]}")
            return agent, 0.01

        with patch.object(type(pool), "_build_one_agent", _flaky_build):
            built = pool.warmup()

        # 2 out of 3 succeed
        assert built == 2
        assert pool.available == 2

    def test_zero_pool_size_warms_nothing(self):
        pool = _make_pool(pool_size=0)
        with _mock_build(pool, num_agents=0):
            built = pool.warmup()
        assert built == 0
        assert pool.available == 0


# ---------------------------------------------------------------------------
# acquire / release - pool path
# ---------------------------------------------------------------------------


class TestAcquireFromPool:
    def test_acquire_returns_pool_agent(self):
        pool = _make_pool(pool_size=2)
        with _mock_build(pool):
            pool.warmup()
        agent, is_overflow = pool.acquire()
        assert agent is not None
        assert is_overflow is False
        assert pool.available == 1  # one fewer

    def test_acquire_decrements_availability(self):
        pool = _make_pool(pool_size=3)
        with _mock_build(pool):
            pool.warmup()
        for expected in (2, 1, 0):
            pool.acquire()
            assert pool.available == expected

    def test_release_increments_availability(self):
        pool = _make_pool(pool_size=2)
        with _mock_build(pool):
            pool.warmup()
        agent, is_overflow = pool.acquire()
        assert pool.available == 1
        pool.release(agent, is_overflow)
        assert pool.available == 2

    def test_stats_track_acquired_released(self):
        pool = _make_pool(pool_size=2)
        with _mock_build(pool):
            pool.warmup()
        agent, is_overflow = pool.acquire()
        pool.release(agent, is_overflow)
        s = pool.stats
        assert s["total_acquired"] == 1
        assert s["total_released"] == 1
        assert s["overflow_total"] == 0


# ---------------------------------------------------------------------------
# acquire - overflow path
# ---------------------------------------------------------------------------


class TestOverflow:
    def test_overflow_when_pool_empty(self):
        pool = _make_pool(pool_size=1)
        with _mock_build(pool):
            pool.warmup()
        # Drain pool
        _a1, _of1 = pool.acquire()
        assert pool.available == 0

        # Overflow path — patch _build_one_agent for this acquire call
        with _mock_build(pool):
            agent2, is_overflow = pool.acquire()

        assert is_overflow is True
        assert pool.stats["overflow_total"] == 1

    def test_overflow_agent_not_returned_to_pool(self):
        pool = _make_pool(pool_size=1)
        with _mock_build(pool):
            pool.warmup()
        _a1, _of1 = pool.acquire()
        with _mock_build(pool):
            agent2, is_overflow = pool.acquire()
        assert is_overflow is True
        pool.release(agent2, is_overflow)
        # Pool should still have 0 (not 1) because overflow was discarded
        assert pool.available == 0

    def test_max_overflow_raises_when_limit_reached(self):
        pool = _make_pool(pool_size=0, max_overflow=1)
        with _mock_build(pool):
            pool.warmup()

        # First overflow — ok
        with _mock_build(pool):
            _a, _of = pool.acquire()

        # Second overflow — should raise
        with _mock_build(pool):
            with pytest.raises(RuntimeError, match="max_overflow"):
                pool.acquire()

    def test_unlimited_overflow_does_not_raise(self):
        pool = _make_pool(pool_size=0, max_overflow=-1)
        with _mock_build(pool):
            pool.warmup()

        agents = []
        with _mock_build(pool):
            for _ in range(20):
                a, of = pool.acquire()
                agents.append((a, of))
        assert pool.stats["overflow_total"] == 20


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_keys_present(self):
        pool = _make_pool(pool_size=2)
        with _mock_build(pool):
            pool.warmup()
        s = pool.stats
        expected_keys = {
            "config_path",
            "pool_size",
            "available",
            "overflow_active",
            "overflow_total",
            "total_acquired",
            "total_released",
            "avg_build_time_s",
        }
        assert expected_keys <= set(s.keys())

    def test_overflow_active_tracks_in_flight(self):
        pool = _make_pool(pool_size=0, max_overflow=-1)
        with _mock_build(pool):
            pool.warmup()

        with _mock_build(pool):
            agent, is_overflow = pool.acquire()
        assert pool.stats["overflow_active"] == 1

        pool.release(agent, is_overflow)
        assert pool.stats["overflow_active"] == 0


# ---------------------------------------------------------------------------
# Thread safety (smoke test)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_acquire_release(self):
        """Many threads acquiring and releasing should not corrupt the pool."""
        pool = _make_pool(pool_size=5)
        with _mock_build(pool):
            pool.warmup()

        errors = []
        call_count = [0]
        lock = threading.Lock()

        def _fake_build_static(self):  # noqa: ANN001
            with lock:
                call_count[0] += 1
                n = call_count[0]
            agent = MagicMock(name=f"overflow_{n}")
            return agent, 0.001

        def worker():
            try:
                with patch.object(type(pool), "_build_one_agent", _fake_build_static):
                    agent, is_overflow = pool.acquire()
                    pool.release(agent, is_overflow)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # All pool agents should be returned eventually
        assert pool.available == 5
