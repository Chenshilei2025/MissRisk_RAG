from __future__ import annotations

import importlib


def test_script_modules_import_without_side_effects() -> None:
    modules = [
        "scripts.baselines.answer_bearing_lexical",
        "scripts.build.mmdocrag",
        "scripts.build.multimodalqa",
        "scripts.build.observation_states",
        "scripts.build.slidevqa",
        "scripts.build.split_by_source",
        "scripts.eval.audit_submission",
        "scripts.eval.predictions",
        "scripts.train.model_b",
        "scripts.train.model_c",
    ]

    for module in modules:
        importlib.import_module(module)

