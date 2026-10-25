"""Tests for the dependency-free command-line interface."""

import pytest

from main import APP_VERSION, _validate_args, build_parser


def test_parser_defaults_to_primary_camera():
    parser = build_parser()
    args = parser.parse_args([])

    assert _validate_args(args, parser) == 0
    assert args.width == 1280
    assert args.height == 720


def test_parser_accepts_named_camera_and_capture_settings():
    parser = build_parser()
    args = parser.parse_args(["--camera", "2", "--width", "640", "--height", "480", "--fps", "24"])

    assert _validate_args(args, parser) == 2
    assert (args.width, args.height, args.fps) == (640, 480, 24)


@pytest.mark.parametrize("arguments", [["-1"], ["--width", "0"], ["--fps", "-1"]])
def test_parser_rejects_invalid_capture_settings(arguments):
    parser = build_parser()
    args = parser.parse_args(arguments)

    with pytest.raises(SystemExit):
        _validate_args(args, parser)


def test_parser_rejects_two_camera_selectors():
    parser = build_parser()
    args = parser.parse_args(["1", "--camera", "2"])

    with pytest.raises(SystemExit):
        _validate_args(args, parser)


def test_version_is_semantic():
    assert APP_VERSION == "1.0.0"
