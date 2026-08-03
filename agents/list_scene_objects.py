"""
Run this with Blocks in Play mode to see the actual mesh/object names
in the scene. We need real names to correctly assign semantic segmentation
IDs — guessing at names (e.g. assuming "Ground" exists) risks silently
mislabeling everything.
"""

import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

objects = client.simListSceneObjects()
print(f"Total objects in scene: {len(objects)}")
print()
for name in sorted(objects):
    print(name)