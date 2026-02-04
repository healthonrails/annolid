from __future__ import annotations

from PIL import Image

from annolid.image_editing.backends.stable_diffusion_cpp_backend import (
    StableDiffusionCppBackend,
)
from annolid.image_editing.types import ImageEditRequest


def test_sdcpp_build_command_includes_inpaint_flags() -> None:
    init = Image.new("RGB", (64, 64), (0, 0, 0))
    mask = Image.new("L", (64, 64), 0)

    req = ImageEditRequest(
        mode="inpaint",
        prompt="remove object",
        negative_prompt="",
        width=64,
        height=64,
        steps=5,
        cfg_scale=3.0,
        seed=123,
        strength=0.5,
        init_image=init,
        mask_image=mask,
        num_images=2,
    )

    cmd = StableDiffusionCppBackend.build_command(
        req,
        sd_cli="/usr/bin/sd-cli",
        ctx_args=[
            "--diffusion-model",
            "/tmp/model.gguf",
            "--vae",
            "/tmp/vae.safetensors",
            "--llm",
            "/tmp/llm.gguf",
        ],
        output="/tmp/out_%03d.png",
        init_img="/tmp/init.png",
        mask_img="/tmp/mask.png",
        ref_imgs=(),
    )

    assert "--diffusion-model" in cmd
    assert "--output" in cmd
    assert "--batch-count" in cmd
    assert "--init-img" in cmd
    assert "--mask" in cmd
    assert "--seed" in cmd
    assert cmd[cmd.index("--batch-count") + 1] == "2"
