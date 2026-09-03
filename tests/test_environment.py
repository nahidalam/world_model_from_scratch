import pytest

from world_models.environment import missing, require_local_inference


def test_missing_reports_absent_packages() -> None:
    assert "definitely_not_a_real_package_xyz" in missing(
        ("definitely_not_a_real_package_xyz",)
    )


def test_missing_returns_empty_for_present_packages() -> None:
    assert missing(("sys",)) == []  # sys is always importable


def test_require_local_inference_raises_when_deps_absent() -> None:
    # Under the artifacts-only profile (no torch/CUDA), the guard must raise a
    # RuntimeError with an actionable message rather than a bare exception.
    try:
        require_local_inference()
    except RuntimeError:
        pass  # expected on an artifacts-only or CPU environment
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"expected RuntimeError, got {type(exc).__name__}: {exc}")
