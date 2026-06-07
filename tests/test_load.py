"""Validate the package loads like ComfyUI loads it (V3 schemas included).

Run from the ComfyUI root with its embedded Python:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\comfy-ui-minecraft-pbr\\tests\\test_load.py
"""

import importlib.util
import os
import sys

COMFY_ROOT = r"M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
PKG_DIR = os.path.join(COMFY_ROOT, "custom_nodes", "comfy-ui-minecraft-pbr")

os.chdir(COMFY_ROOT)
sys.path.insert(0, COMFY_ROOT)

import folder_paths  # noqa: F401,E402  (proves ComfyUI runtime is importable)

# Replicate ComfyUI's custom-node loader: spec_from_file_location with the folder name.
module_name = "comfy-ui-minecraft-pbr"
spec = importlib.util.spec_from_file_location(module_name, os.path.join(PKG_DIR, "__init__.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

mappings = module.NODE_CLASS_MAPPINGS
assert len(mappings) == 7, f"expected 7 nodes, got {len(mappings)}"

for node_id, cls in mappings.items():
    schema = cls.define_schema()
    schema.finalize()  # assigns default output ids, as ComfyUI does before validate()
    schema.validate()
    assert schema.node_id == node_id, f"{node_id}: schema id mismatch ({schema.node_id})"
    n_in = len(schema.inputs)
    n_out = len(schema.outputs)
    print(f"  ok {node_id:26} <{schema.display_name}>  inputs={n_in} outputs={n_out} output_node={schema.is_output_node}")

print("\nNODE PACKAGE LOADS CLEANLY")
