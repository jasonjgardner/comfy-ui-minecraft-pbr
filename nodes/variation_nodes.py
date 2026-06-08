"""Latent variation node: turn one source latent into a batch of N variants.

Model-free (no ``model`` wired): seeded slerp perturbation via the pure-torch
``labpbr.variation`` core. Guided (``model`` wired): img2img re-denoise through
ComfyUI's sampler at ``denoise = variation_strength``. Output is a batched LATENT,
one VAE Decode away from the Pack nodes.
"""

import comfy.samplers
import torch
from comfy_api.latest import io

from ..labpbr.variation import make_variation_batch

CATEGORY = "Minecraft PBR"


def _zero_conditioning(conditioning):
    """Zeroed copy of a CONDITIONING (mirrors ComfyUI's ConditioningZeroOut).

    Used as the default ``negative`` in the guided path when the caller leaves it
    unwired, so img2img has a valid (empty) negative prompt.
    """
    out = []
    for t in conditioning:
        d = t[1].copy()
        pooled = d.get("pooled_output", None)
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t[0]), d])
    return out


class MinecraftLatentVariation(io.ComfyNode):
    """Emit a batch of variant latents from a single source latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MinecraftLatentVariation",
            display_name="Minecraft PBR - Latent Variation",
            category=CATEGORY,
            description="Make N variants of a latent (randomized blocks / weathering). "
            "Model-free seeded slerp by default; prompt-guided img2img when a model is wired.",
            inputs=[
                io.Latent.Input("samples", tooltip="Source latent; if batched, index 0 is varied."),
                io.Int.Input("count", default=4, min=1, max=64, tooltip="Number of variants to emit."),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True, tooltip="Base seed; variant i uses seed + i.",
                ),
                io.Float.Input(
                    "variation_strength", default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip="0 = source, 1 = full divergence. Guided path maps this to denoise.",
                ),
                io.Model.Input("model", optional=True, tooltip="Wire a model to switch to prompt-guided img2img."),
                io.Conditioning.Input("positive", optional=True, tooltip="Required when a model is wired."),
                io.Conditioning.Input("negative", optional=True, tooltip="Defaults to zeroed conditioning."),
                io.Int.Input("steps", default=20, min=1, max=10000, tooltip="Sampler steps (guided path only)."),
                io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, tooltip="CFG scale (guided path only)."),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, tooltip="Sampler (guided path only)."),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, tooltip="Scheduler (guided path only)."),
            ],
            outputs=[io.Latent.Output(display_name="variants")],
        )

    @classmethod
    def execute(
        cls,
        samples,
        count,
        seed,
        variation_strength,
        model=None,
        positive=None,
        negative=None,
        steps=20,
        cfg=7.0,
        sampler_name=None,
        scheduler=None,
    ) -> io.NodeOutput:
        if count < 1:
            raise ValueError("count must be >= 1")

        source = samples["samples"]

        if model is None:
            batch = make_variation_batch(source, seed=seed, count=count, strength=variation_strength)
            return io.NodeOutput({"samples": batch})

        if positive is None:
            raise ValueError("positive conditioning is required when a model is wired")

        from nodes import common_ksampler  # lazy: avoid import cycles at node-load time

        tiled = source[:1].repeat(count, 1, 1, 1)
        neg = negative if negative is not None else _zero_conditioning(positive)
        out = common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, neg, {"samples": tiled}, denoise=variation_strength,
        )[0]
        return io.NodeOutput(out)
