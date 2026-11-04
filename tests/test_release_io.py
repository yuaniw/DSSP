import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import yaml
import zarr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preprocess", ROOT / "policy/DSSP/scripts/process_data.py")
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)


class PreprocessingTests(unittest.TestCase):
    def test_temporal_alignment_and_episode_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_states, expected_actions, expected_clouds = [], [], []
            for episode, length in enumerate([3, 5]):
                states = np.arange(length * 14, dtype=np.float32).reshape(length, 14) + episode * 1000
                clouds = np.arange(length * 4 * 6, dtype=np.float32).reshape(length, 4, 6)
                with h5py.File(root / f"episode{episode}.hdf5", "w") as file:
                    file.create_dataset("joint_action/vector", data=states)
                    file.create_dataset("pointcloud", data=clouds)
                expected_states.append(states[:-1])
                expected_actions.append(states[1:])
                expected_clouds.append(clouds[:-1])
            output = root / "dataset.zarr"
            preprocess.process_demonstrations(root, output, 2)
            result = zarr.open_group(str(output), mode="r")
            np.testing.assert_array_equal(result["data/state"][:], np.concatenate(expected_states))
            np.testing.assert_array_equal(result["data/action"][:], np.concatenate(expected_actions))
            np.testing.assert_array_equal(result["data/point_cloud"][:], np.concatenate(expected_clouds))
            np.testing.assert_array_equal(result["meta/episode_ends"][:], [2, 6])
            self.assertEqual(result["data/point_cloud"].chunks, (6, 4, 6))

    def test_missing_input_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                preprocess.process_demonstrations(root, root / "dataset.zarr", 1)
            self.assertFalse((root / "dataset.zarr").exists())

    def test_existing_dataset_and_symlink_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dataset.zarr"
            output.mkdir()
            sentinel = output / "preserve.txt"
            sentinel.write_text("original")
            link = root / "linked.zarr"
            link.symlink_to(output, target_is_directory=True)
            broken_link = root / "broken.zarr"
            broken_link.symlink_to(root / "missing")
            for path in [output, link, broken_link]:
                with self.assertRaises(FileExistsError):
                    preprocess.process_demonstrations(root, path, 1)
            self.assertEqual(sentinel.read_text(), "original")
            self.assertTrue(link.is_symlink())
            self.assertTrue(broken_link.is_symlink())

    def test_zero_demonstrations_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                preprocess.process_demonstrations(root, root / "dataset.zarr", 0)
            self.assertFalse((root / "dataset.zarr").exists())


class EvaluationArgumentTests(unittest.TestCase):
    def test_overrides_preserve_types_without_executing_python(self):
        # Exercise the CLI parser without loading the simulator or GPU extensions.
        source = ast.parse((ROOT / "script/eval_policy.py").read_text())
        names = {"_normalize_cli_string", "parse_args_and_config"}
        functions = [node for node in source.body if type(node) == ast.FunctionDef and node.name in names]
        import argparse
        namespace = {"argparse": argparse, "yaml": yaml}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "eval_cli", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("task_config: demo_clean" + chr(10) + "ckpt_setting: null" + chr(10))
            sentinel = root / "executed"
            expression = f"__import__('pathlib').Path('{sentinel}').touch()"
            argv = ["eval_policy.py", "--config", str(config), "--overrides",
                    "--eval_video_log", "false", "--eval_episodes", "2",
                    "--checkpoint_path", expression]
            with patch.object(sys, "argv", argv):
                parsed = namespace["parse_args_and_config"]()
            self.assertIs(parsed["eval_video_log"], False)
            self.assertEqual(parsed["eval_episodes"], 2)
            self.assertEqual(parsed["checkpoint_path"], expression)
            self.assertEqual(parsed["ckpt_setting"], "demo_clean")
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
