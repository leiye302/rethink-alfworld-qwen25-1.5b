"""Verify the explicit 1.5B/4-GPU/stop2000 delta against the frozen 3B recipe."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SFT_ALLOWED = {
    "base_model", "base_revision", "parent_dir", "data_dir", "output_dir",
    "epochs", "total_steps", "scheduler_total_steps", "full_resume_checkpoint_steps",
    "prepared_world_size", "minimum_host_memory_gib",
}
OPSD_ALLOWED = {"trainer.n_gpus_per_node", "ray_init.num_gpus",
                "ray_init.address", "ray_init.include_dashboard"}
SOURCE_FILES = {
    "src/sft/route_bce/core.py": "sft/code/route_bce/core.py",
    "src/sft/route_bce/preflight.py": "sft/code/route_bce/preflight.py",
    "src/sft/legacy/stage1_training.py": "sft/code/legacy/stage1_training.py",
    "src/sft/legacy/sft_dataset.py": "sft/code/legacy/sft_dataset.py",
    "src/sft/legacy/rethink_token.py": "sft/code/legacy/rethink_token.py",
    "src/data/build_dataset.py": "data_tools/build_dataset.py",
    "src/data/action_targets.py": "data_tools/action_targets.py",
    "src/data/prompts/hr_review.txt": "prompts/hr_review.txt",
    "src/data/prompts/hr_calibration_examples.txt": "prompts/hr_calibration_examples.txt",
    "src/data/prompts/local_hint_system.txt": "prompts/local_hint_system.txt",
}
SDAR_FILES = [
    "verl/trainer/ppo/core_algos.py", "verl/trainer/ppo/sdar_utils.py",
    "verl/workers/actor/dp_actor.py", "verl/trainer/ppo/rethink_api_protocol.py",
    "verl/trainer/ppo/rethink_local_hints.py", "verl/trainer/ppo/skillsd_ray_trainer.py",
    "agent_system/multi_turn_rollout/rollout_loop.py",
    "agent_system/environments/env_package/alfworld/projection.py",
]


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ported_digest(path):
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    if path.name == "build_dataset.py":
        # The public template moved data_tools/ to src/data/. Undo ONLY those
        # two reviewed path substitutions before comparison to the old source.
        payload = payload.replace(b"ROOT = Path(__file__).resolve().parents[2]",
                                  b"ROOT = Path(__file__).resolve().parents[1]")
        payload = payload.replace(b"default=ROOT/'data/route_sft'", b"default=ROOT/'sft/data_v1'")
    return hashlib.sha256(payload).hexdigest()

def flat(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flat(item, name + "."))
        else:
            result[name] = item
    return result


def delta(old, new):
    return {key: {"reference": old.get(key), "current": new.get(key)}
            for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)}


def snapshot(source):
    """Maintainer-only: emit a portable reference; never copy run logs/secrets."""
    import yaml
    data_lock = json.loads((ROOT / "configs/data.json").read_text())
    for name, wanted in data_lock["files"].items():
        original = (source / "sft/data_v1" / Path(name).name
                    if name.startswith("route_sft/") else source / "data" / name)
        assert hashlib.sha256(original.read_bytes()).hexdigest() == wanted, \
            "Frozen public data differs from the archived 3B data: " + name
    # Only explicitly allowlisted public source/config files are read.
    return {
        "schema": "frozen_3b_recipe_reference_v1",
        "source_data_verified": True,
        "evidence_scope": "Archived public 3B recipe; not a claim of bitwise-identical GPU trajectories.",
        "sft": json.loads((source / "sft/configs/train.json").read_text(encoding="utf-8")),
        "opsd": yaml.safe_load((source / "sdar/verl/trainer/config/rethink_public.yaml").read_text()),
        "source_sha256_lf": {target: digest(source / original) for target, original in SOURCE_FILES.items()},
        "sdar_sha256_lf": {name: digest(source / "sdar" / name) for name in SDAR_FILES},
        "evaluation_manifest_sha256_lf": digest(source / "evaluation/tasks_seen140.json"),
        "evaluation_protocol": json.loads((source / "evaluation/protocol.json").read_text()),
        "frozen_data_lock": data_lock,
    }


def audit(check_framework=False):
    import yaml
    reference = json.loads((ROOT / "configs/reference_3b.json").read_text())
    assert reference["source_data_verified"], "Reference data must be checked against the 3B source"
    sft = json.loads((ROOT / "configs/sft.json").read_text())
    opsd = yaml.safe_load((ROOT / "configs/opsd.yaml").read_text())
    sft_changes = delta(reference["sft"], sft)
    opsd_changes = delta(flat(reference["opsd"]), flat(opsd))
    assert set(sft_changes) <= SFT_ALLOWED, "Unexpected SFT science change: " + repr(set(sft_changes) - SFT_ALLOWED)
    assert set(opsd_changes) <= OPSD_ALLOWED, "Unexpected OPSD science change: " + repr(set(opsd_changes) - OPSD_ALLOWED)
    assert sft["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert sft["base_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert (sft["epochs"], sft["total_steps"], sft["prepared_world_size"]) == (1, 2000, 4)
    assert sft["scheduler_total_steps"] == reference["sft"]["total_steps"] == 4000
    assert sft["full_resume_checkpoint_steps"] == [2000]
    assert opsd["trainer"]["n_gpus_per_node"] == opsd["ray_init"]["num_gpus"] == 4
    assert opsd["ray_init"]["address"] == "local"
    for name, wanted in reference["source_sha256_lf"].items():
        assert ported_digest(ROOT / name) == wanted, "Changed frozen core/data implementation: " + name
    assert json.loads((ROOT / "configs/data.json").read_text()) == reference["frozen_data_lock"]
    assert digest(ROOT / "configs/seen140_tasks.json") == reference["evaluation_manifest_sha256_lf"]
    evaluation = json.loads((ROOT / "configs/eval.json").read_text())
    assert evaluation["task_manifest"] == "seen140_tasks.json"
    evaluation["task_manifest"] = "tasks_seen140.json"  # same hashed list, public filename changed
    assert evaluation == reference["evaluation_protocol"]
    if check_framework:
        for name, wanted in reference["sdar_sha256_lf"].items():
            assert digest(ROOT / ".deps/sdar" / name) == wanted, "Changed SDAR loss/routing code: " + name
    return {"passed": True, "sft_changes": sft_changes, "opsd_changes": opsd_changes,
            "frozen_data_and_core_unchanged": True, "lr_horizon_preserved": True,
            "sdar_core_checked": check_framework, "gpu_runtime_verified": False,
            "boundary": reference["evidence_scope"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", action="store_true")
    p.add_argument("--reference-release", type=Path,
                   help="Maintainer: create a NEW portable reference from a public 3B source release")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.reference_release:
        if not a.out:
            p.error("--reference-release requires a NEW --out file")
        with a.out.open("x", encoding="utf-8") as stream:
            json.dump(snapshot(a.reference_release.resolve()), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print("Portable 3B recipe reference written; no private run material copied.")
        return
    result = audit(a.framework)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
