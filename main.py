import json
import os

import cv2
import numpy as np

from utils import ANNOTATIONS_PATH, CALIB_PATH_RAW, IMAGES_PATH_RAW, PROJECT_ROOT

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "QtAgg"))

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


CAMERAS = (1, 2, 3, 4)
CAM_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
GROUP_COLOR_ORDER = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
GROUP_COLOR_OVERRIDES = {7: "#D000A8"}
XY_FIGURE_PATH = PROJECT_ROOT / "docs" / "groups_xy.png"
CAMERAS_FIGURE_PATH = PROJECT_ROOT / "docs" / "cameras_groups.png"


def group_color(group_id):
    if group_id in GROUP_COLOR_OVERRIDES:
        return to_rgb(GROUP_COLOR_OVERRIDES[group_id])
    return matplotlib.colormaps["tab20"](GROUP_COLOR_ORDER[group_id % len(GROUP_COLOR_ORDER)])[:3]

class ObjectDatas:
    def __init__(self):

        self.cam_id = None
        self.class_name = None
        self.bboxes = None

        self.center_point_2d = []
        self.center_point_3d = []

        self.vertical_axis_image = None
        self.vertical_axis_world = None

    def compute_bbox_vertical_axis(self, bbox, calibration_params):
    
        K = np.array(calibration_params["nK"], dtype=float)
        rt = np.array(calibration_params["rt"], dtype=float)
        
        R = rt[:3, :3]
        t = rt[:3, 3]

        x1, y1, x2, y2 = bbox
    
        # Centro da bounding box
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
    
        center_h = np.array([u, v, 1.0])
    
        # Direção vertical do mundo projetada na imagem
        vertical_direction = np.array([0.0, 0.0, 1.0])
        vanishing_point = K @ R @ vertical_direction
    
        # Reta na imagem: au + bv + c = 0
        line_image = np.cross(center_h, vanishing_point)
    
        norm = np.linalg.norm(line_image[:2])
    
        if norm < 1e-10:
            raise ValueError("Eixo vertical degenerado")
    
        line_image /= norm
    
        # Homografia do plano Z = 0 para a imagem
        H = K @ np.column_stack((
            R[:, 0],
            R[:, 1],
            t
        ))
    
        # Transformar a reta da imagem para o plano XY
        line_world = H.T @ line_image
    
        norm = np.linalg.norm(line_world[:2])
    
        if norm < 1e-10:
            raise ValueError("Reta no plano XY degenerada")
    
        line_world /= norm
    
        return line_image, line_world

    def get_ann_data(self, ann, calibration_params, cam_id, z=0.0):

        self.cam_id = cam_id
        self.class_name = ann["class_name"]

        bbox = ann["bbox"]

        self.boxes = np.array([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])

        u = (bbox["x1"] + bbox["x2"]) / 2
        v = bbox["y2"]

        self.center_point_2d = np.array([u, v])

        K = calibration_params["nK"]
        T_camera_world = calibration_params["rt_inv"]
        ray_camera = np.linalg.solve(K, np.array([u, v, 1.0]))
        origin_world = T_camera_world[:3, 3]
        direction_world = T_camera_world[:3, :3] @ ray_camera

        scale = (z - origin_world[2]) / direction_world[2]

        point3d = origin_world + scale * direction_world

        self.center_point_3d = point3d

        self.vertical_axis_image, self.vertical_axis_world = self.compute_bbox_vertical_axis(self.boxes, calibration_params)


def draw_world_axis(ax, obj, color, length=2.0):

    a, b, c = obj.vertical_axis_world

    # Ponto XY estimado anteriormente
    point = obj.center_point_3d[:2]

    # Ponto da reta mais próximo da estimativa XY
    normal = np.array([a, b])

    distance = a * point[0] + b * point[1] + c

    center = point - distance * normal

    # Vetor diretor da reta
    direction = np.array([-b, a])

    p1 = center - (length / 2) * direction
    p2 = center + (length / 2) * direction

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=color,
        linewidth=1.5,
        alpha=0.7
    )

def annotations_to_object(annotations, calibs_cams):

    objects = []
    for idx, ann in enumerate(annotations):
        for obj in ann:

            obj_data = ObjectDatas()

            obj_data.get_ann_data(obj, calibs_cams[idx], idx+1)

            objects.append(obj_data)
    
    return objects    

def carregar_calibracao(camera_id):
    path = CALIB_PATH_RAW / f"calib_rt{camera_id}.npz"
    return np.load(path)

def carregar_anotacoes(camera_id):
    path = ANNOTATIONS_PATH / f"camera_{camera_id}.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)["annotations"]

def draw_image_axis(image, obj, group_id):
    a, b, c = obj.vertical_axis_image
    x1, y1, x2, y2 = obj.boxes
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    height, width = image.shape[:2]
    x1, x2 = np.clip((x1, x2), 0, width - 1)
    y1, y2 = np.clip((y1, y2), 0, height - 1)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    red, green, blue = group_color(group_id)
    color = tuple(round(channel * 255) for channel in (blue, green, red))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"G{group_id}"
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_top = max(0, y1 - text_size[1] - baseline - 4)
    cv2.rectangle(image, (x1, label_top), (x1 + text_size[0] + 6, label_top + text_size[1] + baseline + 4), color, -1)
    cv2.putText(image, label, (x1 + 3, label_top + text_size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    intersections = []

    if abs(b) > 1e-10:
        for x in (x1, x2):
            y = int(round((-a * x - c) / b))
            if y1 <= y <= y2:
                intersections.append((x, y))

    if abs(a) > 1e-10:
        for y in (y1, y2):
            x = int(round((-b * y - c) / a))
            if x1 <= x <= x2:
                intersections.append((x, y))

    if len(intersections) >= 2:
        cv2.line(image, intersections[0], intersections[1], color, 2)


def distance_matrix(objects):

    points_xy = np.array([
        obj.center_point_3d[:2]
        for obj in objects
    ])

    diff = points_xy[:, None, :] - points_xy[None, :, :]

    distances = np.linalg.norm(diff, axis=2)

    return distances

    
def intersect_world_axes(
    obj1,
    obj2,
    max_offset=0.20,
    min_angle_deg=12.0
):
    if obj1.cam_id == obj2.cam_id:
        return None

    l1 = obj1.vertical_axis_world
    l2 = obj2.vertical_axis_world

    A = np.array([l1[:2], l2[:2]])
    b = -np.array([l1[2], l2[2]])

    # Como as normais estão normalizadas,
    # |det(A)| representa |sin(ângulo entre as retas)|
    sin_angle = abs(np.linalg.det(A))

    if sin_angle < np.sin(np.deg2rad(min_angle_deg)):
        return None

    intersection = np.linalg.solve(A, b)

    if not np.all(np.isfinite(intersection)):
        return None

    p1 = obj1.center_point_3d[:2]
    p2 = obj2.center_point_3d[:2]

    d1 = np.linalg.norm(intersection - p1)
    d2 = np.linalg.norm(intersection - p2)

    # Interseção precisa ser plausível para ambas as detecções
    if d1 > max_offset or d2 > max_offset:
        return None

    return intersection

from itertools import combinations


def group_objects(
    chairs,
    max_offset=0.20,
    min_angle_deg=12.0,
    max_intersection_spread=0.5
):
    n = len(chairs)

    # Inicialmente, cada detecção é um grupo
    groups = [[i] for i in range(n)]

    pairs = []
    intersections = {}

    # 1. Construir candidatos usando interseções
    for i, j in combinations(range(n), 2):

        obj_i = chairs[i]
        obj_j = chairs[j]

        intersection = intersect_world_axes(
            obj_i,
            obj_j,
            max_offset=max_offset,
            min_angle_deg=min_angle_deg
        )

        if intersection is None:
            continue

        p1 = obj_i.center_point_3d[:2]
        p2 = obj_j.center_point_3d[:2]

        d1 = np.linalg.norm(intersection - p1)
        d2 = np.linalg.norm(intersection - p2)

        # Custo geométrico da associação
        cost = d1 + d2

        pairs.append((cost, i, j))
        intersections[(i, j)] = intersection

    # 2. Priorizar associações com menor custo
    pairs.sort(key=lambda item: item[0])

    for cost, i, j in pairs:

        group_i = next(g for g in groups if i in g)
        group_j = next(g for g in groups if j in g)

        if group_i is group_j:
            continue

        # 3. Impedir câmeras repetidas
        cams_i = {chairs[k].cam_id for k in group_i}
        cams_j = {chairs[k].cam_id for k in group_j}

        if not cams_i.isdisjoint(cams_j):
            continue

        candidate_group = group_i + group_j

        # 4. Verificar todas as interseções do grupo proposto
        group_intersections = []
        valid = True

        for a, b in combinations(sorted(candidate_group), 2):

            point = intersections.get((a, b))

            if point is None:
                valid = False
                break

            group_intersections.append(point)

        if not valid:
            continue

        # 5. Verificar se as interseções convergem
        points = np.array(group_intersections)

        center = np.median(points, axis=0)

        errors = np.linalg.norm(points - center, axis=1)

        if np.max(errors) > max_intersection_spread:
            continue

        # 6. Unir os grupos
        group_i.extend(group_j)
        groups.remove(group_j)

    groups.sort(key=lambda g: min(g))

    return groups

def estimate_group_position(group, chairs):

    if len(group) == 1:
        return chairs[group[0]].center_point_3d[:2].copy()

    lines = np.array([
        chairs[i].vertical_axis_world
        for i in group
    ])

    A = lines[:, :2]
    b = -lines[:, 2]

    position, _, _, _ = np.linalg.lstsq(
        A, b, rcond=None
    )

    return position


def plot_xy_groups(ax, chairs, groups, show_detection_indices=True):
    for group_id, group in enumerate(groups):
        color = group_color(group_id)

        for index in group:
            obj = chairs[index]
            x, y = obj.center_point_3d[:2]
            draw_world_axis(ax, obj, color=color, length=2.0)
            ax.scatter(x, y, color=color, marker=CAM_MARKERS[obj.cam_id], s=45, edgecolors="black", linewidths=0.4, zorder=3)
            if show_detection_indices:
                ax.annotate(str(index), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

        x, y = estimate_group_position(group, chairs)
        ax.scatter(x, y, marker="x", color=color, s=120, linewidths=2.5, zorder=5)
        ax.annotate(
            f"G{group_id}", (x, y), xytext=(8, -17), textcoords="offset points",
            color=color, fontsize=9, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.9}
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Posições XY e grupos de cadeiras")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)

    for cam_id in CAMERAS:
        ax.scatter([], [], marker=CAM_MARKERS[cam_id], color="gray", edgecolors="black", linewidths=0.4, label=f"Câmera {cam_id}")
    ax.scatter([], [], marker="x", color="black", label="Posição do grupo")
    ax.legend(loc="upper left", fontsize=8)

def main():

    calibs_cams = []
    for cam in CAMERAS:
        calib = carregar_calibracao(cam)
        calibs_cams.append(calib)

    annotations = []
    for cam in CAMERAS:
        ann = carregar_anotacoes(cam)
        annotations.append(ann)

    objects = annotations_to_object(annotations, calibs_cams)

    camera_images = {}
    for cam in CAMERAS:
        image = cv2.imread(str(IMAGES_PATH_RAW / f"camera_{cam}.png"))
        if image is not None:
            camera_images[cam] = image

    chairs = [
        obj for obj in objects
        if obj.class_name == "chair"
    ]

    D = distance_matrix(chairs)
    
    groups = group_objects(
        chairs,
        max_offset=1.2,
        min_angle_deg=12.0,
        max_intersection_spread=0.5
    )

    for group_id, group in enumerate(groups):
        for index in group:
            obj = chairs[index]
            if obj.cam_id in camera_images:
                draw_image_axis(camera_images[obj.cam_id], obj, group_id)
    
    print(f"Detecções: {len(chairs)}")
    print(f"Grupos encontrados: {len(groups)}")
    
    for group_id, group in enumerate(groups):
    
        position = estimate_group_position(group, chairs)
    
        cameras = [chairs[i].cam_id for i in group]
    
        print(
            f"Grupo {group_id}: "
            f"detecções={group}, "
            f"câmeras={cameras}, "
            f"XY={np.round(position, 3)}"
        )

    plot_xy_groups(ax1, chairs, groups)

    xy_fig, xy_ax = plt.subplots(figsize=(10, 8))
    plot_xy_groups(xy_ax, chairs, groups, show_detection_indices=False)
    xy_fig.tight_layout()
    XY_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    xy_fig.savefig(XY_FIGURE_PATH, dpi=180)
    print(f"Gráfico XY salvo em: {XY_FIGURE_PATH}")

    # Gráfico 2: matriz de distâncias
    im = ax2.imshow(D, cmap="viridis")

    ax2.set_xlabel("Índice do objeto")
    ax2.set_ylabel("Índice do objeto")
    ax2.set_title("Matriz de distâncias")

    fig.colorbar(im, ax=ax2, label="Distância (m)")

    plt.tight_layout()

    fig_images, image_axes = plt.subplots(2, 2, figsize=(14, 8))
    for cam, ax in zip(CAMERAS, image_axes.flat):
        if cam in camera_images:
            ax.imshow(cv2.cvtColor(camera_images[cam], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Câmera {cam}: cadeiras por grupo")
        ax.axis("off")

    fig_images.tight_layout()
    CAMERAS_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig_images.savefig(CAMERAS_FIGURE_PATH, dpi=180)
    print(f"Imagens das câmeras salvas em: {CAMERAS_FIGURE_PATH}")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()

if __name__ == "__main__":
    main()
