"""
Position probe — continuously prints the drone's live position while you fly
it manually (keyboard/RC) in a precompiled environment like AirSimNH, where
there's no Editor viewport to click-and-read coordinates from.

Run AirSimNH.sh first, THEN run this script in a separate terminal. Do NOT
call enableApiControl(True) anywhere else while this is running, or manual
keyboard control will be overridden and you won't be able to fly.

Fly around the map (default AirSim keyboard controls: arrow keys / WASD
depending on build, space to ascend, C to descend — check AirSim's docs if
these don't respond) and watch the printed coordinates to note down good
candidate start/goal positions and roughly where the open, obstacle-free
areas are.

Press Ctrl+C to stop.

Usage:
    python position_probe.py
"""

import airsim
import time


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    # Deliberately NOT calling enableApiControl(True) — leaving manual/RC
    # control active so you can fly around freely while this just observes.

    print("Polling position every 0.5s. Fly manually and note coordinates "
          "for spots you want to use as start/goal. Ctrl+C to stop.\n")

    try:
        while True:
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(orientation)

            collision = client.simGetCollisionInfo()
            collision_note = "  <- COLLIDING" if collision.has_collided else ""

            print(f"x={pos.x_val:7.2f}  y={pos.y_val:7.2f}  z={pos.z_val:7.2f}  "
                  f"yaw={yaw:5.2f}rad{collision_note}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()