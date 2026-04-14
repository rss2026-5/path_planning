#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from scipy.spatial.transform import Rotation as R


# =========================
# CONFIG
# =========================
bag_path = "bag_files/sim_rrt_run_5"   # folder, NOT .db3 file
ODOM_TOPIC = "/odom"
PATH_TOPIC = "/planned_trajectory/path"


# =========================
# LOAD TYPESTORE
# =========================
typestore = get_typestore(Stores.ROS2_HUMBLE)  # change if needed

class TFBufferOffline:
    """
    Minimal offline TF buffer for rosbag data.
    Stores transforms: parent -> child
    """

    def __init__(self):
        self.transforms = {}  # (parent, child) -> (t, x, y, yaw)

    def add_transform(self, parent, child, x, y, z, q):
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        yaw = rot.as_euler("xyz")[2]

        self.transforms[(parent, child)] = (x, y, yaw)

    def transform_point(self, x, y, from_frame, to_frame):
        """
        ONLY handles map <-> odom style transforms (2D simplified)
        """
        if (to_frame, from_frame) in self.transforms:
            tx, ty, tyaw = self.transforms[(to_frame, from_frame)]

            # rotate + translate
            c, s = np.cos(tyaw), np.sin(tyaw)

            xr = c * x - s * y + tx
            yr = s * x + c * y + ty
            return xr, yr

        if (from_frame, to_frame) in self.transforms:
            tx, ty, tyaw = self.transforms[(from_frame, to_frame)]

            c, s = np.cos(-tyaw), np.sin(-tyaw)

            x -= tx
            y -= ty

            xr = c * x - s * y
            yr = s * x + c * y
            return xr, yr

        raise RuntimeError("TF transform missing between frames")


# =========================================================
# LOAD TF FROM BAG
# =========================================================

def load_tf(reader):
    tf_buffer = TFBufferOffline()

    tf_topics = [c for c in reader.connections if "tf" in c.topic]

    for conn, timestamp, rawdata in reader.messages(connections=tf_topics):
        msg = reader.deserialize(rawdata, conn.msgtype)

        for t in msg.transforms:
            p = t.header.frame_id
            c = t.child_frame_id
            tr = t.transform.translation
            q = t.transform.rotation

            tf_buffer.add_transform(p, c, tr.x, tr.y, tr.z, q)

    return tf_buffer


# # =========================================================
# # EXTRACT ODOM (in map frame via TF)
# # =========================================================

# def extract_odom_map(reader, tf_buffer):
#     x, y = [], []

#     connections = [c for c in reader.connections if c.topic == ODOM_TOPIC]

#     for conn, timestamp, rawdata in reader.messages(connections=connections):
#         msg = reader.deserialize(rawdata, conn.msgtype)

#         ox = msg.pose.pose.position.x
#         oy = msg.pose.pose.position.y

#         # IMPORTANT: convert base_link → map
#         try:
#             mx, my = tf_buffer.transform_point(
#                 ox, oy,
#                 from_frame="base_link",
#                 to_frame="map"
#             )
#         except:
#             # fallback if TF missing
#             mx, my = ox, oy

#         x.append(mx)
#         y.append(my)

#     return np.array(x), np.array(y)

# def extract_path_map(reader, tf_buffer):
#     ref_x, ref_y = [], []

#     connections = [c for c in reader.connections if c.topic == PATH_TOPIC]

#     for conn, timestamp, rawdata in reader.messages(connections=connections):
#         msg = reader.deserialize(rawdata, conn.msgtype)

#         if hasattr(msg, "points"):
#             for p in msg.points:

#                 # assume path is in "odom" or "base_link"
#                 try:
#                     mx, my = tf_buffer.transform_point(
#                         p.x, p.y,
#                         from_frame=msg.header.frame_id,
#                         to_frame="map"
#                     )
#                 except:
#                     mx, my = p.x, p.y

#                 ref_x.append(mx)
#                 ref_y.append(my)

#         break

#     return np.array(ref_x), np.array(ref_y)


# =========================
# EXTRACT ODOM
# =========================

def extract_odom(reader):
    x, y, yaw, t = [], [], [], []

    connections = [c for c in reader.connections if c.topic == ODOM_TOPIC]

    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, conn.msgtype)

        x.append(msg.pose.pose.position.x)
        y.append(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        yaw.append(quaternion_to_yaw(q))

        t.append(timestamp * 1e-9)

    return np.array(x), np.array(y), np.array(yaw), np.array(t)


# =========================
# EXTRACT PATH
# =========================
def extract_path(reader):
    ref_x, ref_y = [], []

    connections = [c for c in reader.connections if c.topic == PATH_TOPIC]

    if not connections:
        raise RuntimeError(f"❌ Topic '{PATH_TOPIC}' not found")

    for conn in connections:
        print(f"Using topic: {conn.topic} ({conn.msgtype})")

    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, conn.msgtype)

        # Handle Marker message
        if hasattr(msg, "points") and len(msg.points) > 0:
            for p in msg.points:
                ref_x.append(p.x)
                ref_y.append(p.y)
        else:
            raise RuntimeError("❌ Marker has no points")

        break  # assume one trajectory message

    return np.array(ref_x), np.array(ref_y)

# ======
# other stuff
# =======

def quaternion_to_yaw(q):
    # ROS quaternion → yaw
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

def compute_path_heading(ref_x, ref_y):
    dx = np.diff(ref_x)
    dy = np.diff(ref_y)

    headings = np.arctan2(dy, dx)

    # pad to match length
    headings = np.append(headings, headings[-1])

    return headings

def nearest_path_index(ax, ay, ref_x, ref_y):
    dists = np.sqrt((ref_x - ax)**2 + (ref_y - ay)**2)
    return np.argmin(dists)

def compute_heading_error(x, y, yaw, ref_x, ref_y):
    path_heading = compute_path_heading(ref_x, ref_y)

    errors = []

    for ax, ay, theta in zip(x, y, yaw):
        idx = nearest_path_index(ax, ay, ref_x, ref_y)
        theta_path = path_heading[idx]

        err = theta - theta_path

        # normalize to [-pi, pi]
        err = (err + np.pi) % (2 * np.pi) - np.pi

        errors.append(err)

    return np.array(errors)

# =========================
# METRICS
# =========================
def compute_cte(actual_x, actual_y, ref_x, ref_y):
    errors = []
    for ax, ay in zip(actual_x, actual_y):
        dists = np.sqrt((ref_x - ax)**2 + (ref_y - ay)**2)
        errors.append(np.min(dists))
    return np.array(errors)


def path_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


# =========================
# MAIN
# =========================
def main():
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:

        # print("\nAvailable topics:")
        # for c in reader.connections:
        #     print(f"  {c.topic} ({c.msgtype})")

        # Extract data
        # x, y, t = extract_odom(reader)
        x, y, yaw, t = extract_odom(reader)
        ref_x, ref_y = extract_path(reader)

        # print("\nLoading TF tree...")
        # tf_buffer = load_tf(reader)

        # print("Extracting odom in map frame...")
        # x, y = extract_odom_map(reader, tf_buffer)

        # print("Extracting path in map frame...")
        # ref_x, ref_y = extract_path_map(reader, tf_buffer)

    # =========================
    # METRICS
    # =========================
    # x, y = align_by_centroid(x, y, ref_x, ref_y)
    cte = compute_cte(x, y, ref_x, ref_y)
    heading_error = compute_heading_error(x, y, yaw, ref_x, ref_y)

    print("\n=== TRACKING METRICS ===")
    print(f"Mean CTE: {np.mean(cte):.4f} m")
    print(f"Max CTE:  {np.max(cte):.4f} m")
    print(f"RMSE:     {np.sqrt(np.mean(cte**2)):.4f} m")

    print("\n=== PATH METRICS ===")
    print(f"Planned path length: {path_length(ref_x, ref_y):.2f} m")
    print(f"Actual path length:  {path_length(x, y):.2f} m")

    print("\n=== HEADING METRICS ===")
    print(f"Mean heading error: {np.mean(np.abs(heading_error)):.4f} rad")
    print(f"Max heading error:  {np.max(np.abs(heading_error)):.4f} rad")
    print(f"RMSE:               {np.sqrt(np.mean(heading_error**2)):.4f} rad")

    # =========================
    # PLOTS
    # =========================

    # Trajectory overlay
    plt.figure()
    plt.plot(ref_x, ref_y, '--', label='Planned Path')
    plt.plot(x, y, label='Actual Path')
    plt.legend()
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Tracking')
    plt.axis('equal')

    # TF trajectory overlay
    # plt.figure()
    # plt.plot(ref_x, ref_y, '--', label="Planned Path (map)")
    # plt.plot(x, y, label="Robot Trajectory (map)")

    # plt.axis("equal")
    # plt.legend()
    # plt.title("Trajectory Overlay (TF-Aligned)")
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")

    # Error vs time
    plt.figure()
    plt.plot(t, cte)
    plt.xlabel('Time (s)')
    plt.ylabel('Cross-Track Error (m)')
    plt.title('Tracking Error vs Time')

    # heading error vs time
    plt.figure()
    plt.plot(t, heading_error)
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Error (rad)')
    plt.title('Heading Error vs Time')
    plt.show()

    plt.show()


if __name__ == "__main__":
    main()