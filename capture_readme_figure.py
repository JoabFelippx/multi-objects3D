"""Capture a matched frame from four cameras and draw the README experiment."""

import argparse
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_wire.core import Channel, Subscription
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import object_matching as matching


OUTPUT_PATH = Path(__file__).resolve().parent / "docs" / "readme_experiment.png"
CAMERAS = matching.CAMERAS
MAX_CACHED_FRAMES = 120
MAX_REFERENCE_DISTANCE = 0.75


@dataclass(frozen=True)
class Reference:
    name: str
    class_name: str
    xy: tuple[float, float]
    height: float


REFERENCES = (
    Reference("C1", "chair", (0.0, 0.0), 0.970),
    Reference("C2", "chair", (1.0, 1.0), 0.920),
    Reference("C3", "chair", (-1.0, 1.0), 0.920),
    Reference("C4", "chair", (-1.0, -1.0), 0.815),
    Reference("P1", "person", (1.0, -1.0), 1.810),
)


def remember(cache, sequence, value):
    cache[sequence] = value
    cache.move_to_end(sequence)
    while len(cache) > MAX_CACHED_FRAMES:
        cache.popitem(last=False)


def capture_matched_frames(broker_uri, timeout_seconds):
    """Keep only camera frames with the same sequence as their detections."""
    channel = Channel(uri=broker_uri)
    subscription = Subscription(channel)
    for cam_id in CAMERAS:
        subscription.subscribe(f"CameraGateway.{cam_id}.Frame")
        subscription.subscribe(f"ObjectDetector.{cam_id}.Detection")

    images = {cam_id: OrderedDict() for cam_id in CAMERAS}
    detections = {cam_id: OrderedDict() for cam_id in CAMERAS}
    captured = {}
    deadline = time.monotonic() + timeout_seconds

    try:
        while len(captured) < len(CAMERAS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                message = channel.consume(timeout=min(remaining, 1.0))
            except TimeoutError:
                continue

            parts = message.topic.split(".")
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            cam_id = int(parts[1])
            if cam_id not in images or cam_id in captured:
                continue

            if parts[0] == "CameraGateway" and parts[2] == "Frame":
                image = message.unpack(Image)
                if not image.data:
                    continue
                sequence = int(image.sequence)
                remember(images[cam_id], sequence, bytes(image.data))
            elif parts[0] == "ObjectDetector" and parts[2] == "Detection":
                annotation = message.unpack(ObjectAnnotations)
                sequence = int(annotation.frame_id)
                remember(detections[cam_id], sequence, annotation)
            else:
                continue

            if sequence not in images[cam_id] or sequence not in detections[cam_id]:
                continue

            encoded = np.frombuffer(images[cam_id][sequence], dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                images[cam_id].pop(sequence)
                continue

            captured[cam_id] = (frame, detections[cam_id][sequence], sequence)
            print(f"Câmera {cam_id}: imagem e detecção do frame {sequence}")
    finally:
        channel.close()

    missing = sorted(set(CAMERAS) - set(captured))
    if missing:
        raise TimeoutError(
            f"Sem imagem e detecção do mesmo frame nas câmeras {missing}"
        )
    return captured


def assign_references(clusters):
    """Associate ground truth and estimates one to one within each class."""
    assignments = {}
    for class_name in {reference.class_name for reference in REFERENCES}:
        reference_ids = [
            index
            for index, reference in enumerate(REFERENCES)
            if reference.class_name == class_name
        ]
        group_ids = [
            index
            for index, group in enumerate(clusters)
            if group.class_name == class_name
        ]
        if not reference_ids or not group_ids:
            continue

        true_xy = np.array([REFERENCES[index].xy for index in reference_ids])
        estimated_xy = np.array(
            [clusters[index].estimation_point for index in group_ids]
        )
        distances = np.linalg.norm(
            true_xy[:, None, :] - estimated_xy[None, :, :], axis=2
        )
        unmatched_costs = np.full(
            (len(reference_ids), len(reference_ids)), MAX_REFERENCE_DISTANCE
        )
        costs = np.column_stack((distances, unmatched_costs))
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if (
                column < len(group_ids)
                and distances[row, column] <= MAX_REFERENCE_DISTANCE
            ):
                assignments[reference_ids[row]] = group_ids[column]
    return assignments


def group_colors(clusters):
    palette = plt.get_cmap("tab20")
    order = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    return {index: palette(order[index % len(order)]) for index in range(len(clusters))}


def detection_groups(objects, clusters):
    result = {}
    for index, group in enumerate(clusters):
        for object_index in group.object_indices:
            previous = result.get(object_index)
            if previous is None:
                result[object_index] = index
                continue
            point = objects[object_index].center_point_3d[:2]
            old_distance = np.linalg.norm(point - clusters[previous].estimation_point)
            new_distance = np.linalg.norm(point - group.estimation_point)
            if new_distance < old_distance:
                result[object_index] = index
    return result


def draw_camera(ax, cam_id, frame, objects, object_groups, colors, visible_groups):
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Câmera {cam_id}", fontsize=13, weight="bold")
    ax.axis("off")

    for index, obj in enumerate(objects):
        if obj.cam_id != cam_id:
            continue
        group_id = object_groups.get(index)
        if group_id not in visible_groups:
            continue
        color = colors[group_id]
        x1, y1, x2, y2 = obj.bbox_raw
        x1 = max(0, float(x1))
        y1 = max(0, float(y1))
        x2 = min(frame.shape[1] - 1, float(x2))
        y2 = min(frame.shape[0] - 1, float(y2))
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.5,
            )
        )
        label = f"G{group_id + 1} · {obj.class_name}"
        ax.text(
            x1,
            max(12, y1 - 6),
            label,
            color="white",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 2.5},
        )


def draw_world(ax, clusters, assignments, colors):
    ax.set_title("Posição no mundo · Z = 0", fontsize=14, weight="bold")
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(alpha=0.25)
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.axvline(0, color="#888888", linewidth=0.7)

    for reference_id, reference in enumerate(REFERENCES):
        group_id = assignments.get(reference_id)
        color = colors[group_id] if group_id is not None else "#777777"
        real = np.array(reference.xy)
        ax.scatter(
            *real,
            marker="o",
            s=110,
            facecolors="white",
            edgecolors=[color],
            linewidths=2.5,
            zorder=4,
        )
        ax.annotate(
            f"{reference.name} real",
            real,
            xytext=(5, 7),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )

    reverse = {group_id: ref_id for ref_id, group_id in assignments.items()}
    for group_id, group in enumerate(clusters):
        if group_id not in reverse:
            continue
        estimated = group.estimation_point
        color = colors[group_id]
        ax.scatter(*estimated, marker="x", s=120, color=color, linewidths=2.5, zorder=5)
        ax.annotate(
            f"G{group_id + 1} {group.class_name}",
            estimated,
            xytext=(6, -14),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )
        if group_id in reverse:
            real = np.array(REFERENCES[reverse[group_id]].xy)
            ax.plot(
                [real[0], estimated[0]],
                [real[1], estimated[1]],
                "--",
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )

    ax.plot(
        [],
        [],
        "o",
        markerfacecolor="white",
        markeredgecolor="black",
        label="Posição real",
    )
    ax.plot([], [], "x", color="black", label="Posição estimada")
    ax.legend(loc="upper right", fontsize=8)


def draw_errors(xy_ax, height_ax, clusters, assignments, colors):
    names = [reference.name for reference in REFERENCES]
    y = np.arange(len(REFERENCES))
    xy_errors = []
    height_errors = []
    for ref_id, reference in enumerate(REFERENCES):
        group_id = assignments.get(ref_id)
        if group_id is None:
            xy_errors.append(None)
            height_errors.append(None)
            continue
        group = clusters[group_id]
        xy_errors.append(float(np.linalg.norm(group.estimation_point - reference.xy)))
        height_errors.append(float(group.height - reference.height))

    xy_ax.set_title("Erro de posição XY", fontsize=13, weight="bold")
    xy_ax.set_yticks(y, names)
    xy_ax.set_ylim(len(REFERENCES) - 0.5, -0.5)
    xy_ax.set_xlabel("Distância até a posição real (m)")
    xy_ax.set_xlim(
        0, max(0.35, *(error * 1.35 for error in xy_errors if error is not None))
    )
    xy_ax.grid(axis="x", alpha=0.2)
    for ref_id, error in enumerate(xy_errors):
        group_id = assignments.get(ref_id)
        if error is None:
            xy_ax.text(0.01, y[ref_id], "sem grupo", va="center", color="#777777")
            continue
        xy_ax.barh(y[ref_id], error, height=0.55, color=colors[group_id])
        xy_ax.text(error + 0.01, y[ref_id], f"{error:.3f} m", va="center", fontsize=9)

    height_ax.set_title("Altura real e estimada", fontsize=13, weight="bold")
    height_ax.set_yticks(y, names)
    height_ax.set_ylim(len(REFERENCES) - 0.5, -0.5)
    height_ax.set_xlabel("Altura (m)")
    height_ax.set_xlim(
        0,
        max(
            2.15,
            *(clusters[group_id].height * 1.2 for group_id in assignments.values()),
        ),
    )
    height_ax.grid(axis="x", alpha=0.2)
    for ref_id, reference in enumerate(REFERENCES):
        group_id = assignments.get(ref_id)
        height_ax.scatter(
            reference.height,
            y[ref_id],
            marker="o",
            s=55,
            facecolors="white",
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )
        if group_id is None:
            height_ax.text(
                reference.height + 0.05,
                y[ref_id],
                "sem grupo",
                va="center",
                color="#777777",
                fontsize=9,
            )
            continue
        estimated = clusters[group_id].height
        height_ax.plot(
            [reference.height, estimated],
            [y[ref_id], y[ref_id]],
            color=colors[group_id],
            linewidth=2,
        )
        height_ax.scatter(
            estimated,
            y[ref_id],
            marker="x",
            s=65,
            color=colors[group_id],
            linewidths=2,
            zorder=5,
        )
        height_ax.text(
            max(reference.height, estimated) + 0.05,
            y[ref_id],
            f"{height_errors[ref_id]:+.3f} m",
            va="center",
            fontsize=9,
        )
    height_ax.plot(
        [], [], "o", markerfacecolor="white", markeredgecolor="black", label="Real"
    )
    height_ax.plot([], [], "x", color="black", label="Estimado")
    height_ax.legend(loc="upper right", fontsize=8)


def build_figure(captured, objects, clusters, assignments, output_path):
    colors = group_colors(clusters)
    object_groups = detection_groups(objects, clusters)
    visible_groups = set(assignments.values())

    figure = plt.figure(figsize=(18, 15), constrained_layout=True)
    layout = figure.add_gridspec(3, 2, height_ratios=(1, 1, 1.1))
    for index, cam_id in enumerate(CAMERAS):
        ax = figure.add_subplot(layout[index // 2, index % 2])
        draw_camera(
            ax,
            cam_id,
            captured[cam_id][0],
            objects,
            object_groups,
            colors,
            visible_groups,
        )

    world_ax = figure.add_subplot(layout[2, 0])
    error_layout = layout[2, 1].subgridspec(2, 1)
    xy_ax = figure.add_subplot(error_layout[0, 0])
    height_ax = figure.add_subplot(error_layout[1, 0])
    draw_world(world_ax, clusters, assignments, colors)
    draw_errors(xy_ax, height_ax, clusters, assignments, colors)

    figure.suptitle(
        "Quatro câmeras · cadeiras e pessoa · posições reais × estimadas",
        fontsize=19,
        weight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Tempo máximo para encontrar os quatro pares de frames",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="Arquivo PNG de saída"
    )
    args = parser.parse_args()

    matching.class_target = ["chair", "person"]
    calibrations = [matching.carregar_calibracao(cam_id) for cam_id in CAMERAS]
    captured = capture_matched_frames(matching.BROKER_URI, args.timeout)
    annotations = {cam_id: captured[cam_id][1] for cam_id in CAMERAS}
    objects = matching.message_to_objects(annotations, calibrations, matching.GROUND_Z)
    clusters = matching.match_objects(objects)
    assignments = assign_references(clusters)
    build_figure(captured, objects, clusters, assignments, args.output)

    print(f"Imagem salva em {args.output}")
    for ref_id, reference in enumerate(REFERENCES):
        group_id = assignments.get(ref_id)
        if group_id is None:
            print(f"{reference.name} ({reference.class_name}): sem grupo")
            continue
        group = clusters[group_id]
        xy_error = np.linalg.norm(group.estimation_point - reference.xy)
        height_error = group.height - reference.height
        print(
            f"{reference.name}: G{group_id + 1}, erro XY = {xy_error:.3f} m, erro de altura = {height_error:+.3f} m"
        )


if __name__ == "__main__":
    main()
