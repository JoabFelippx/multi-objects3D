import os
import socket
from pathlib import Path

import cv2
import numpy as np

from is_wire.core import Channel, Message, Subscription
from is_msgs.image_pb2 import ObjectAnnotations
from sklearn.cluster import DBSCAN

CAMERAS = [1, 2, 3, 4]
class_target = ["chair"]
CALIB_PATH_RAW = Path(
    os.environ.get(
        "CALIB_PATH_RAW", Path(__file__).resolve().parent / "data" / "calibration"
    )
)
BROKER_URI = os.environ.get("BROKER_URI", "amqp://guest:guest@10.10.50.176:30000")
GROUND_Z = 0.0
PUBLISHED_Z = 0.01
MIN_CROSSING_ANGLE_DEG = 20
MAX_INTERSECTION_DISTANCE = 0.50
CLUSTER_EPS = 0.25
CLUSTER_MIN_SAMPLES = 2
MAX_HEIGHT_REPROJECTION_ERROR = 6.65


class StreamChannel(Channel):
    def consume_last(self, timeout=None):
        try:
            message = super().consume(timeout=timeout)
        except socket.timeout:
            return False

        while True:
            try:
                message = super().consume(timeout=0.0)
            except socket.timeout:
                return message


def carregar_calibracao(camera_id):
    path = CALIB_PATH_RAW / f"calib_rt{camera_id}.npz"

    with np.load(path) as calib_file:
        params = {key: calib_file[key] for key in calib_file.files}

    return Camera(params)


class StreamHandler:
    def __init__(self, broker_uri, cameras):
        self.broker_uri = broker_uri
        self.cameras = cameras

        self.object_channels = {}
        self._subs = {}

        for cam_id in self.cameras:
            channel = StreamChannel(self.broker_uri)

            subscription = Subscription(channel)
            subscription.subscribe(topic=f"ObjectDetector.{cam_id}.Detection")

            self.object_channels[cam_id] = channel
            self._subs[cam_id] = subscription

    def _consume(self, cam_id):
        channel = self.object_channels[cam_id]

        msg = channel.consume_last()

        if isinstance(msg, bool):
            return None

        return msg.unpack(ObjectAnnotations)

    def get_messages(self):
        messages = {}

        for cam_id in self.cameras:
            annotations = self._consume(cam_id)

            messages[cam_id] = annotations

        return messages


class Camera:
    def __init__(self, calib_params):
        self.camera_id = calib_params["cam_id"]

        self.K = calib_params["K"]
        self.nK = calib_params["nK"]

        self.dist = calib_params["dist"]

        self.T_world_camera = calib_params["rt"]
        self.R_world_camera = calib_params["rt"][:3, :3]
        self.Trans_world_camera = calib_params["rt"][:3, 3]

        self.T_camera_world = calib_params["rt_inv"]
        self.R_camera_world = calib_params["rt_inv"][:3, :3]
        self.Trans_camera_world = calib_params["rt_inv"][:3, 3]

        self.P = calib_params["P_original"]
        self.P_undistorted = calib_params["P_undistorted"]

        self.height = calib_params["height"]
        self.width = calib_params["width"]

        self.roi = calib_params["roi"]

    def projection_pixel_in_ground_plane(self, u, v, plane_z=0.0, is_undistorted=True):
        if is_undistorted:
            point_homogeneous = np.array([u, v, 1.0], dtype=float)
            ray_camera_sys = np.linalg.inv(self.nK) @ point_homogeneous
        else:
            point = np.array([[[u, v]]], dtype=np.float64)
            undistorted = cv2.undistortPoints(point, self.K, self.dist)
            x, y = undistorted[0, 0]
            ray_camera_sys = np.array([x, y, 1.0])

        origin_world = self.Trans_camera_world
        direction_world = self.R_camera_world @ ray_camera_sys

        dz = direction_world[2]

        lambda_scale = (plane_z - origin_world[2]) / dz
        point_world = origin_world + lambda_scale * direction_world

        return point_world

    def world_to_pixel(self, point_3D, is_undistorted=True):
        point_3D = np.asarray(point_3D, dtype=float).reshape(3)

        pt_cam = self.R_world_camera @ point_3D + self.Trans_world_camera

        if is_undistorted:
            pt_img = self.nK @ pt_cam

            u = pt_img[0] / pt_img[2]
            v = pt_img[1] / pt_img[2]

        else:
            rvec, _ = cv2.Rodrigues(self.R_world_camera)

            projected, _ = cv2.projectPoints(
                point_3D.reshape(1, 1, 3),
                rvec,
                self.Trans_world_camera.reshape(3, 1),
                self.K,
                self.dist,
            )

            u, v = projected[0, 0]

        return np.array([u, v], dtype=float)

    def get_homography(self, plane_z=0.0, is_undistorted=True):
        K = self.nK if is_undistorted else self.K

        rt_inv = np.linalg.inv(self.T_world_camera)

        rt_inv[2, 3] += plane_z

        rt = np.linalg.inv(rt_inv)

        R = rt[:3, :3]
        t = rt[:3, 3]

        H = K @ np.column_stack((R[:, 0], R[:, 1], t))

        return H

    def estimate_z_from_pixel(self, u, v, X, Y):
        K = self.nK
        R = self.R_world_camera
        t = np.asarray(self.Trans_world_camera).reshape(3)

        point_world = np.array([X, Y, 0.0])
        a = K @ (R @ point_world + t)
        b = K @ R[:, 2]

        A = np.array([u * b[2] - b[0], v * b[2] - b[1]])

        B = np.array([a[0] - u * a[2], a[1] - v * a[2]])

        if np.linalg.norm(A) < 1e-10:
            return None

        Z, _, _, _ = np.linalg.lstsq(A.reshape(-1, 1), B, rcond=None)

        Z = float(Z[0])

        # Verificar a reprojeção
        point_3d = np.array([X, Y, Z])

        point_camera = R @ point_3d + t

        if point_camera[2] <= 0:
            return None

        projected = K @ point_camera

        projected = projected[:2] / projected[2]

        error = np.linalg.norm(projected - np.array([u, v]))

        return Z, error


class ObjectDatas:
    def __init__(
        self,
    ):
        self.cam_id = None
        self.class_name = None
        self.bboxe = []

        self.center_point_2d = []
        self.center_point_3d = []

        self.top_point = []

        self.vertical_axis_image = None
        self.vertical_axis_world = None

        self.camera = None


class DetectionsGroup:
    def __init__(self):
        self.class_name = None
        self.estimation_point = None
        self.points = None
        self.pairs = None
        self.object_indices = None
        self.camera_ids = None
        self.ambiguous = None
        self.height = None


def get_point_to_projection(x1, y1, x2, y2):
    u = (x1 + x2) / 2
    v = max(y1, y2)

    return np.array([u, v])


def get_top_point(x1, y1, x2, y2):
    u = (x1 + x2) / 2
    v = min(y1, y2)

    return np.array([u, v])


def undistort_point(vertices, K, dist, nK):
    distorted_points = np.array(
        [[vertex.x, vertex.y] for vertex in vertices],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    points = cv2.undistortPoints(
        distorted_points,
        np.asarray(K, dtype=float),
        np.asarray(dist, dtype=float),
        P=np.asarray(nK, dtype=float),
    ).reshape(-1, 2)

    return points


def get_vanishing_point(direction, K, R):
    return K @ R @ direction


def message_to_objects(messages, cameras, z_plane=0.0):
    objects = []

    for cam_id, message in messages.items():
        if message is None or isinstance(message, bool):
            continue

        for detection in message.objects:
            class_name = detection.label
            if class_name not in class_target:
                continue

            vertices = detection.region.vertices
            if not vertices:
                continue

            obj = ObjectDatas()

            cam_obj = cameras[cam_id - 1]

            undistorted_points = undistort_point(
                vertices, cam_obj.K, cam_obj.dist, cam_obj.nK
            )

            x1 = np.min(undistorted_points[:, 0])
            y1 = np.min(undistorted_points[:, 1])
            x2 = np.max(undistorted_points[:, 0])
            y2 = np.max(undistorted_points[:, 1])

            to_projection = get_point_to_projection(x1, y1, x2, y2)
            point_in_ground_plane = cam_obj.projection_pixel_in_ground_plane(
                to_projection[0], to_projection[1], z_plane
            )

            top_point = get_top_point(x1, y1, x2, y2)

            obj.cam_id = cam_id
            obj.class_name = class_name
            obj.bboxe = np.array([x1, y1, x2, y2])

            obj.center_point_2d = to_projection
            obj.center_point_3d = point_in_ground_plane

            obj.top_point = top_point

            vanishing_point = get_vanishing_point(
                np.array([0.0, 0.0, 1.0]), cam_obj.nK, cam_obj.R_world_camera
            )

            point_homogeneous = np.array([to_projection[0], to_projection[1], 1.0])
            line_image = np.cross(point_homogeneous, vanishing_point)

            norm = np.linalg.norm(line_image[:2])

            if norm < 1e-10:
                raise ValueError("Eixo vertical degenerado")

            line_image /= norm

            H = cam_obj.get_homography()

            line_world = H.T @ line_image

            norm = np.linalg.norm(line_world[:2])

            if norm < 1e-10:
                raise ValueError("Reta no plano XY degenerada")

            line_world /= norm

            obj.vertical_axis_image = line_image
            obj.vertical_axis_world = line_world

            obj.camera = cam_obj

            objects.append(obj)

    return objects


def intersection_world_axes(l1, l2, p1, p2):
    A = np.array([l1[:2], l2[:2]])
    b = -np.array([l1[2], l2[2]])

    # Como as normais estão normalizadas,
    # |det(A)| representa |sin(ângulo entre as retas)|
    sin_angle = abs(np.linalg.det(A))

    if sin_angle < np.sin(np.deg2rad(MIN_CROSSING_ANGLE_DEG)):
        return None

    intersection = np.linalg.solve(A, b)

    if not np.all(np.isfinite(intersection)):
        return None

    d1 = np.linalg.norm(intersection - p1)
    d2 = np.linalg.norm(intersection - p2)

    # Interseção precisa ser plausível para ambas as detecções
    if d1 > MAX_INTERSECTION_DISTANCE or d2 > MAX_INTERSECTION_DISTANCE:
        return None

    return intersection


def intersection_matrix(objects):
    N = len(objects)

    intersection_mtx = np.full((N, N, 2), np.nan)

    for idx1, obj1 in enumerate(objects):
        for idx2 in range(idx1 + 1, N):
            obj2 = objects[idx2]

            # Não comparar objetos da mesma câmera
            if obj1.cam_id == obj2.cam_id:
                continue

            # Não comparar objetos de classes diferentes
            if obj1.class_name != obj2.class_name:
                continue

            l1 = obj1.vertical_axis_world
            l2 = obj2.vertical_axis_world

            p1 = obj1.center_point_3d[:2]
            p2 = obj2.center_point_3d[:2]

            intersection = intersection_world_axes(l1, l2, p1, p2)

            if intersection is None:
                continue

            if not np.all(np.isfinite(intersection)):
                continue

            intersection_mtx[idx1, idx2] = intersection
            intersection_mtx[idx2, idx1] = intersection

    return intersection_mtx


def cluster_intersections(intersection_mtx, objects, eps=0.15, min_samples=2):
    N = len(objects)
    clusters = []

    for class_name in class_target:
        points = []
        pairs = []

        for i in range(N):
            if objects[i].class_name != class_name:
                continue

            for j in range(i + 1, N):
                if objects[j].class_name != class_name:
                    continue

                if objects[i].cam_id == objects[j].cam_id:
                    continue

                intersection = intersection_mtx[i, j]

                if not np.all(np.isfinite(intersection)):
                    continue

                points.append(intersection)
                pairs.append((i, j))

        if not points:
            continue

        points = np.asarray(points)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        labels = dbscan.fit_predict(points)

        for label in np.unique(labels):
            # -1 representa ruído
            if label == -1:
                continue

            mask = labels == label

            cluster_points = points[mask]

            cluster_pairs = [pairs[i] for i in np.flatnonzero(mask)]

            # Recuperar os índices das detecções
            object_indices = sorted({idx for pair in cluster_pairs for idx in pair})

            camera_ids = [objects[idx].cam_id for idx in object_indices]

            # Mais de uma detecção da mesma câmera
            # indica possível mistura de objetos
            ambiguous = len(camera_ids) != len(set(camera_ids))

            group_obj = DetectionsGroup()

            group_obj.class_name = class_name
            group_obj.points = cluster_points
            group_obj.pairs = cluster_pairs
            group_obj.object_indices = object_indices
            group_obj.camera_ids = camera_ids
            group_obj.ambiguous = ambiguous

            clusters.append(group_obj)

    return clusters


def estimate_cluster_position(cluster, objects):
    object_indices = cluster.object_indices

    lines = np.array([objects[idx].vertical_axis_world for idx in object_indices])

    A = lines[:, :2]
    b = -lines[:, 2]

    if np.linalg.matrix_rank(A) < 2:
        return None

    position, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return position


def estimate_cluster_height(cluster, objects, max_error=MAX_HEIGHT_REPROJECTION_ERROR):
    if cluster.estimation_point is None:
        return None

    X, Y = cluster.estimation_point

    heights = []

    for idx in cluster.object_indices:
        obj = objects[idx]

        u, v = obj.top_point

        result = obj.camera.estimate_z_from_pixel(u, v, X, Y)

        if result is None:
            continue

        Z, error = result

        if not np.isfinite(Z) or not np.isfinite(error):
            continue

        if error > max_error:
            continue

        if Z < 0:
            continue

        heights.append(Z)

    if not heights:
        return None

    return float(np.median(heights))


def grupos_para_anotacoes_3d(clusters):
    annotations = ObjectAnnotations()

    for group_id, cluster in enumerate(clusters):
        position_xy = cluster.estimation_point
        position_z = PUBLISHED_Z

        annotation = annotations.objects.add()
        annotation.id = group_id + 1
        annotation.label = cluster.class_name

        keypoint = annotation.keypoints.add()
        keypoint.position.x = float(position_xy[0])
        keypoint.position.y = float(position_xy[1])
        keypoint.position.z = position_z

    return annotations


def main():
    calibs_cam = []
    for cam_id in CAMERAS:
        calibs_cam.append(carregar_calibracao(cam_id))

    stream = StreamHandler(broker_uri=BROKER_URI, cameras=CAMERAS)

    publisher = StreamChannel(BROKER_URI)

    while True:
        messages = stream.get_messages()
        objects = message_to_objects(messages, calibs_cam, GROUND_Z)
        matrix = intersection_matrix(objects)

        clusters = cluster_intersections(
            matrix, objects, eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES
        )

        valid_clusters = []

        for idx, cluster in enumerate(clusters):
            if cluster.ambiguous:
                continue

            position = estimate_cluster_position(cluster, objects)

            if position is None:
                continue

            cluster.estimation_point = position

            height = estimate_cluster_height(cluster, objects)

            if height is None:
                continue

            cluster.height = height

            valid_clusters.append(cluster)

            print(f"\nCluster {idx}")
            print(f"Classe: {cluster.class_name}")
            print(f"Posição XY: {cluster.estimation_point}")
            print(f"Altura Z: {cluster.height:.3f} m")
            print(f"Interseções: {len(cluster.points)}")
            print(f"Detecções: {cluster.object_indices}")
            print(f"Câmeras: {cluster.camera_ids}")
            print(f"Ambíguo: {cluster.ambiguous}")

        output_annotations = grupos_para_anotacoes_3d(valid_clusters)
        output_message = Message()
        output_message.topic = "SkeletonDetector.3D.Annotations"
        output_message.pack(output_annotations)
        publisher.publish(output_message)


if __name__ == "__main__":
    main()
