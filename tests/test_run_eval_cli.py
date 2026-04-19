import pytest

from app.evals.run_eval import _build_parser, _validate_planner_flags


def _parse_args(argv: list[str]):
    parser = _build_parser()
    args = parser.parse_args(argv)
    return parser, args


def test_force_planner_requires_planner_enabled_or_compare_mode():
    parser, args = _parse_args(["--force-planner"])

    with pytest.raises(SystemExit):
        _validate_planner_flags(parser, args)


def test_force_planner_allowed_with_planner_enabled_mode():
    parser, args = _parse_args(["--planner-enabled", "--force-planner"])
    _validate_planner_flags(parser, args)


def test_force_planner_allowed_with_compare_mode():
    parser, args = _parse_args(["--compare-planner", "--force-planner"])
    _validate_planner_flags(parser, args)
