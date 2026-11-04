# Third-party code notices

DSSP includes code derived from the following projects. Their license and copyright notices are retained with the source.

- **RoboTwin:** simulator tasks, robot integration, and collection/evaluation infrastructure. The original MIT license and copyright notice are retained in [LICENSE](LICENSE). The recorded base revision is `ea25643210c3036e57eb44b4cd4b59b8dc1e98d7`.
- **Diffusion Policy and 3D Diffusion Policy:** policy interfaces, training/checkpoint infrastructure, normalization, replay-buffer utilities, diffusion scheduling, EMA, and point-cloud policy components under `policy/DSSP/`. See the Columbia Artificial Intelligence and Robotics Lab MIT notice in [licenses/diffusion-policy-MIT.txt](licenses/diffusion-policy-MIT.txt).
- **External packages:** Mamba-SSM, causal-conv1d, PyTorch, PyTorch3D, Diffusers, SAPIEN, and other installed packages retain their respective licenses. Their packages and simulator assets are not bundled by this source release.

Project references: [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), and [Mamba](https://github.com/state-spaces/mamba).

The root [LICENSE](LICENSE) contains RoboTwin's retained MIT notice.
