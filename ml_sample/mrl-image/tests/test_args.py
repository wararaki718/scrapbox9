import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.args import parse_args


class ArgsTests(unittest.TestCase):
    def test_parse_args_converts_dimensions_to_integers(self) -> None:
        args = parse_args(["--embedding-dim", "16", "--dimensions", "4,8,16"])

        self.assertEqual(args.dimensions, [4, 8, 16])

    def test_parse_args_uses_unit_loss_weights_by_default(self) -> None:
        args = parse_args(["--dimensions", "4,8"])

        self.assertEqual(args.loss_weights, [1.0, 1.0])

    def test_parse_args_parses_loss_weights(self) -> None:
        args = parse_args(["--dimensions", "4,8", "--loss-weights", "2,0.5"])

        self.assertEqual(args.loss_weights, [2.0, 0.5])

    def test_parse_args_rejects_invalid_loss_weights(self) -> None:
        cases = (
            ("", "loss-weights must not be empty"),
            ("1", "loss-weights must contain one value per dimension"),
            ("1,-0.5", "loss-weights must be non-negative finite numbers"),
            ("nan,1", "loss-weights must be non-negative finite numbers"),
            ("inf,1", "loss-weights must be non-negative finite numbers"),
            ("nope,1", "loss-weights must be comma-separated numbers"),
        )

        for value, message in cases:
            with self.subTest(value=value), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()) as error_output:
                parse_args(["--dimensions", "4,8", "--loss-weights", value])

            self.assertIn(message, error_output.getvalue())

    def test_parse_args_uses_default_sample_counts(self) -> None:
        args = parse_args([])

        self.assertEqual(args.train_samples, 1000)
        self.assertEqual(args.test_samples, 200)

    def test_parse_args_rejects_non_positive_sample_counts(self) -> None:
        for option in ("--train-samples", "--test-samples"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_args([option, "0"])

    def test_parse_args_rejects_duplicate_dimensions(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--dimensions", "8,8"])

    def test_parse_args_rejects_non_positive_configuration(self) -> None:
        for option in ("--epochs", "--batch-size", "--embedding-dim", "--learning-rate"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_args([option, "0"])

    def test_parse_args_rejects_non_finite_learning_rates(self) -> None:
        for argv in (
            ["--learning-rate", "nan"],
            ["--learning-rate", "inf"],
            ["--learning-rate=-inf"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_parse_args_rejects_malformed_or_empty_dimensions(self) -> None:
        for value in ("8,invalid", ""):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_args(["--dimensions", value])

    def test_parse_args_rejects_non_positive_prefix_dimensions(self) -> None:
        for argv in (
            ["--dimensions", "0,8"],
            ["--dimensions=-1,8"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_parse_args_rejects_dimensions_exceeding_embedding_dim(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--embedding-dim", "16", "--dimensions", "8,32"])

    def test_parse_args_rejects_malformed_device(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--device", "cpus"])

    def test_parse_args_accepts_valid_device_syntax(self) -> None:
        args = parse_args(["--device", "cpu"])

        self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()