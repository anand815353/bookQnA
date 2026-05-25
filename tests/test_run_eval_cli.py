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


def test_default_output_path_planner_comparison_prefix():
    from app.evals.run_eval import _default_output_path

    p = _default_output_path(compare_planner=True)
    assert p.parent.name == "reports"
    assert p.name.startswith("planner_comparison_")
    assert p.suffix == ".json"


def test_default_output_path_single_run_prefix():
    from app.evals.run_eval import _default_output_path

    p = _default_output_path(compare_planner=False)
    assert p.name.startswith("eval_report_")
