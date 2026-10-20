"""Explicit, non-overlapping GPU allocation; no device discovery or mutation."""
import argparse
import os
import re


def devices(text, count):
    values = [v.strip() for v in text.split(",")]
    if (len(values) != count or len(set(values)) != count
            or any(not re.fullmatch(r"(?:0|[1-9]\d*|GPU-[A-Za-z0-9-]+)", v) for v in values)):
        raise ValueError(f"Expected {count} distinct whole GPU IDs (not MIG/duplicates)")
    return values


def layout(env=None):
    env = os.environ if env is None else env
    baseline = devices(env.get("BASELINE_GPUS", "0,1,2,3"), 4)
    method = devices(env.get("METHOD_GPUS", "4,5,6,7"), 4)
    if set(baseline) & set(method):
        raise ValueError("Baseline and method GPU allocations overlap")
    if len({v.startswith("GPU-") for v in baseline + method}) != 1:
        raise ValueError("Use either physical GPU indices or UUIDs consistently")
    return {"baseline": baseline, "method": method}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["baseline", "sft", "opsd"], required=True)
    a = p.parse_args()
    groups = layout()
    print(",".join(groups["baseline" if a.stage == "baseline" else "method"]))


if __name__ == "__main__":
    main()
