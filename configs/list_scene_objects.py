"""
Lists every scene object in the currently-running AirSim map and groups them
by common name prefix, to help figure out what mesh-name patterns to use in
a new environment's segmentation_classes config.

Run this with the target map (e.g. City) already loaded in Unreal.

Usage:
    python list_scene_objects.py
"""

import re
import os
from collections import defaultdict
from datetime import datetime

import airsim


def guess_prefix(name):
    """
    Groups names like 'Cylinder2', 'Cylinder_2', 'Cylinder8' under 'Cylinder',
    so you get a manageable list of families instead of hundreds of individual
    instance names. Strips trailing digits/underscores.
    """
    return re.sub(r'[_\d]+$', '', name)


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    objects = client.simListSceneObjects()
    print(f"Total scene objects: {len(objects)}\n")

    groups = defaultdict(list)
    for name in objects:
        groups[guess_prefix(name)].append(name)

    # sort by group size, descending — the biggest groups are usually the
    # ones worth turning into a segmentation class (e.g. lots of "Building_*"
    # instances), while one-off names are probably lights/cameras/UI actors
    sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    for prefix, names in sorted_groups:
        example = names[0] if len(names) == 1 else f"{names[0]}, {names[1]}" if len(names) > 1 else ""
        print(f"{prefix:30s} x{len(names):4d}   e.g. {example}")

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(log_dir, f"scene_objects_{run_id}.txt")
    with open(out_path, "w") as f:
        f.write(f"Total scene objects: {len(objects)}\n\n")
        for prefix, names in sorted_groups:
            f.write(f"--- {prefix} (x{len(names)}) ---\n")
            for n in names:
                f.write(f"  {n}\n")

    print(f"\nFull list (including every individual name) saved to: {out_path}")
    print("\nNext step: decide which prefixes correspond to real obstacles vs "
          "background/non-physical scene elements (cameras, lights, sky, UI), "
          "then fill those regex patterns into configs/city.yaml's segmentation_classes.")


if __name__ == "__main__":
    main() 