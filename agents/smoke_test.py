import airsim
import numpy as np
import time

# Step 1: connection
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print("Connected. API control enabled.")

# Step 2: takeoff + send action, read state back
client.takeoffAsync().join()
client.moveByVelocityAsync(1, 0, 0, duration=2).join()

state = client.getMultirotorState()
pos = state.kinematics_estimated.position
vel = state.kinematics_estimated.linear_velocity
orientation = state.kinematics_estimated.orientation

print(f"Position: {pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f}")
print(f"Velocity: {vel.x_val:.2f}, {vel.y_val:.2f}, {vel.z_val:.2f}")
print(f"Orientation (quaternion): {orientation}")

# Step 3: collision detection
collision_info = client.simGetCollisionInfo()
print(f"Has collided: {collision_info.has_collided}")

# Step 4: goal distance placeholder
goal = np.array([10, 0, -5])
current = np.array([pos.x_val, pos.y_val, pos.z_val])
dist = np.linalg.norm(goal - current)
print(f"Distance to goal: {dist:.2f}")

client.takeoffAsync().join()
time.sleep(1)  # let things settle
print(f"Has collided after takeoff settle: {client.simGetCollisionInfo().has_collided}")