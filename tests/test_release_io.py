import ast
import importlib.util
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import imageio_ffmpeg
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


class VideoRecordingTests(unittest.TestCase):
    def test_video_encodes_and_decodes_with_bundled_ffmpeg(self):
        module = runpy.run_path(str(ROOT / "envs/utils/images_to_video.py"))
        frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
        frames[:, :, :, 0] = 160
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode0.mp4"
            module["images_to_video"](frames, str(output), fps=10)
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output),
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True)
        self.assertEqual(len(result.stdout), frames.size)

    def test_failed_evaluation_encoder_raises(self):
        source = ast.parse((ROOT / "envs/_base_task.py").read_text())
        class MethodFinder(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if node.name == "_del_eval_video_ffmpeg":
                    self.method = node

        finder = MethodFinder()
        finder.visit(source)
        method = finder.method
        namespace = {"subprocess": subprocess}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "video_cleanup", "exec"), namespace)
        task = MagicMock()
        task.eval_video_ffmpeg.wait.return_value = 1
        task.eval_video_ffmpeg.args = ["ffmpeg", "episode0.mp4"]
        with self.assertRaises(subprocess.CalledProcessError):
            namespace["_del_eval_video_ffmpeg"](task)
        task.eval_video_ffmpeg.stdin.close.assert_called_once()


class ExternalServiceConfigurationTests(unittest.TestCase):
    def test_training_disables_external_tracking_by_default(self):
        config = yaml.safe_load((ROOT / "policy/DSSP/src/dssp/config/dssp.yaml").read_text())
        self.assertEqual(config["logging"]["mode"], "disabled")

    def test_code_generation_uses_explicit_credentials_and_provider_endpoints(self):
        sdk = MagicMock()
        sdk.OpenAI.return_value.chat.completions.create.return_value.choices[0].message.content = "generated"
        with patch.dict(sys.modules, {"openai": sdk}):
            module = runpy.run_path(str(ROOT / "code_gen/gpt_agent.py"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai", "DEEPSEEK_API_KEY": "test-deepseek"}, clear=True):
            self.assertEqual(module["generate"]([], gpt="openai"), "generated")
            sdk.OpenAI.assert_called_with(api_key="test-openai")
            module["generate"]([], gpt="deepseek")
            sdk.OpenAI.assert_called_with(api_key="test-deepseek", base_url="https://api.deepseek.com")
        sdk.OpenAI.reset_mock()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(KeyError):
            module["generate"]([], gpt="openai")
        sdk.OpenAI.assert_not_called()

    def test_azure_description_service_requires_user_configuration(self):
        inference = MagicMock()
        credentials = MagicMock()
        modules = {"azure.ai.inference": inference, "azure.ai.inference.models": MagicMock(),
                   "azure.core.credentials": credentials}
        settings = {"AZURE_API_KEY": "test-azure",
                    "AZURE_INFERENCE_ENDPOINT": "https://example.invalid/model",
                    "AZURE_INFERENCE_MODEL": "test-model"}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, settings, clear=True):
            module = runpy.run_path(str(ROOT / "description/utils/agent.py"))
        credentials.AzureKeyCredential.assert_called_once_with("test-azure")
        inference.ChatCompletionsClient.assert_called_once_with(
            endpoint=settings["AZURE_INFERENCE_ENDPOINT"],
            credential=credentials.AzureKeyCredential.return_value)
        self.assertEqual(module["model_name"], "test-model")

    def test_image_analysis_reads_credentials_from_environment(self):
        sdk = MagicMock()
        sdk.OpenAI.return_value.chat.completions.create.return_value.choices[0].message.content = "observed"
        with patch.dict(sys.modules, {"openai": sdk, "gpt_agent": MagicMock()}):
            module = runpy.run_path(str(ROOT / "code_gen/observation_agent.py"))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"KIMI_API_KEY": "test-moonshot"}, clear=True):
            task_dir = Path(directory) / "task"
            task_dir.mkdir()
            (task_dir / "episode0_step.png").write_bytes(b"test image")
            result = module["observe_task_execution"](0, "task",
                {"description": "test", "goal": "test"}, save_dir=directory)
        self.assertEqual(result, "observed")
        sdk.OpenAI.assert_called_once_with(api_key="test-moonshot", base_url="https://api.moonshot.cn/v1")


if __name__ == "__main__":
    unittest.main()
