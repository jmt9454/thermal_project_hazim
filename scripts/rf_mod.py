import json
import math
import numpy as np
from pathlib import Path


def unreal_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)

    # Unreal Engine:
    # X = forward
    # Y = right
    # Z = up
    forward = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch)
    ])

    right = np.array([
        -math.sin(yaw),
        math.cos(yaw),
        0.0
    ])

    up = np.cross(forward, right)

    # Apply roll
    if abs(roll) > 1e-8:
        right_rolled = (
            right * math.cos(roll)
            + up * math.sin(roll)
        )

        up_rolled = (
            -right * math.sin(roll)
            + up * math.cos(roll)
        )

        right = right_rolled
        up = up_rolled

    return np.column_stack((forward, right, up))


def world_to_camera(point_world, camera_location, camera_rotation):
    cam_pos = np.array([
        camera_location["x"],
        camera_location["y"],
        camera_location["z"]
    ], dtype=float)

    point = np.array(point_world, dtype=float)

    R = unreal_rotation_matrix(
        camera_rotation["pitch"],
        camera_rotation["yaw"],
        camera_rotation["roll"]
    )

    relative = point - cam_pos

    # Convert world-space position into camera space
    return R.T @ relative


def check_in_frame(
    point_world,
    camera_location,
    camera_rotation,
    horizontal_fov_deg=90.0,
    width=1920,
    height=1080
):
    p = world_to_camera(
        point_world,
        camera_location,
        camera_rotation
    )

    depth = p[0]
    right = p[1]
    up = p[2]

    if depth <= 0:
        return False, None, None

    aspect = width / height

    hfov = math.radians(horizontal_fov_deg)

    vfov = 2.0 * math.atan(
        math.tan(hfov / 2.0) / aspect
    )

    horizontal_limit = depth * math.tan(hfov / 2.0)
    vertical_limit = depth * math.tan(vfov / 2.0)

    in_frame = (
        abs(right) <= horizontal_limit
        and abs(up) <= vertical_limit
    )

    ndc_x = right / horizontal_limit
    ndc_y = up / vertical_limit

    screen_x = (ndc_x + 1.0) / 2.0
    screen_y = (1.0 - ndc_y) / 2.0

    # Convert numpy types to normal Python types
    return (
        bool(in_frame),
        float(screen_x),
        float(screen_y)
    )


# ==========================================================
# SETTINGS
# ==========================================================

HORIZONTAL_FOV = 90.0   # CHANGE to your actual camera FOV
IMAGE_WIDTH = 1920      # CHANGE to your render width
IMAGE_HEIGHT = 1080     # CHANGE to your render height


# ==========================================================
# PROCESS ALL JSON FILES IN CURRENT DIRECTORY
# ==========================================================

current_directory = Path(".")

json_files = sorted(current_directory.glob("*.json"))

processed_count = 0
error_count = 0

print(f"Found {len(json_files)} JSON files.\n")


for json_path in json_files:

    # Prevent processing files previously created by this script
    if json_path.stem.endswith("_with_rf_frame"):
        continue

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Make sure this looks like one of our frame JSON files
        if "camera_location" not in data:
            print(f"Skipping {json_path.name}: no camera_location")
            continue

        if "camera_rotation" not in data:
            print(f"Skipping {json_path.name}: no camera_rotation")
            continue

        if "rf_sources" not in data:
            print(f"Skipping {json_path.name}: no rf_sources")
            continue

        camera_location = data["camera_location"]
        camera_rotation = data["camera_rotation"]

        # --------------------------------------------------
        # Process every RF source in this frame
        # --------------------------------------------------

        for rf in data["rf_sources"]:

            rf_position = [
                rf["world_x"],
                rf["world_y"],
                rf["world_z"]
            ]

            in_frame, screen_x, screen_y = check_in_frame(
                rf_position,
                camera_location,
                camera_rotation,
                horizontal_fov_deg=HORIZONTAL_FOV,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT
            )

            rf["in_frame"] = bool(in_frame)

            if screen_x is not None:
                rf["screen_x"] = float(screen_x)
                rf["screen_y"] = float(screen_y)
            else:
                rf["screen_x"] = None
                rf["screen_y"] = None

        # --------------------------------------------------
        # Save modified JSON
        # --------------------------------------------------

        output_path = json_path.with_name(
            json_path.stem + "_with_rf_frame.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        processed_count += 1

        print(
            f"Processed: {json_path.name}"
            f" -> {output_path.name}"
        )

    except Exception as e:
        error_count += 1
        print(f"ERROR processing {json_path.name}: {e}")


print("\n===================================")
print("Finished")
print(f"Processed: {processed_count}")
print(f"Errors:    {error_count}")
print("===================================")