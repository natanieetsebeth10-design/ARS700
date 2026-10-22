"""
make_worlds.py - Generates the Webots worlds for Component B.

Run:
    python make_worlds.py
    python make_worlds.py --webots R2025a

Creates three worlds in ../worlds/:
    swarm_distributed.wbt - Robots allocate sectors using distributed PSO.
    swarm_centralised.wbt - An elected leader allocates sectors.
    swarm_collect.wbt - Robots collect real IR sensor readings.

All three worlds use the same arena, obstacles, sector markers and
four e-puck robots. Only the controller arguments differ.

The arena layout comes from swarm_core.py so the Webots worlds and
Python experiments use the same geometry.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "swarm_agent"))

import swarm_core as sc  # noqa: E402


PROTO_URL = ("https://raw.githubusercontent.com/cyberbotics/webots/{ver}/"
             "projects/robots/gctronic/e-puck/protos/E-puck.proto")


def look_at_orientation(pos, target=(0, 0, 0)):
    """Calculate the Webots camera orientation for a given position."""
    pos, target = np.array(pos, float), np.array(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.column_stack([r, u, -f])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= np.linalg.norm(axis)
    return axis, angle


def build_world(version, mode):
    year = int("".join(ch for ch in version[1:5] if ch.isdigit()) or 2023)
    L = [f"#VRML_SIM {version} utf8"]
    if year >= 2023:
        L.append(f'EXTERNPROTO "{PROTO_URL.format(ver=version)}"')
    L.append("")
    L += ["WorldInfo {", "  title \"Component B - multi-robot swarm\"", "  basicTimeStep 32",
          "  coordinateSystem \"ENU\"", "}"]
    axis, ang = look_at_orientation((0.0, -1.7, 2.9))
    L += ["Viewpoint {", f"  orientation {axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f} {ang:.4f}",
          "  position 0.0 -1.7 2.9", "}"]
    L += ["Background {", "  skyColor [ 0.85 0.9 0.95 ]", "}",
          "DirectionalLight {", "  direction 0.3 0.6 -1", "  intensity 1.3", "}",
          "DirectionalLight {", "  direction -0.4 -0.3 -1", "  intensity 0.6", "}"]

    # Create the arena floor.
    L += ["Solid {", "  translation 0 0 -0.005", "  children [",
          "    Shape {", "      appearance PBRAppearance { baseColor 0.82 0.82 0.78 roughness 1 metalness 0 }",
          "      geometry Box { size 2 2 0.01 }", "    }", "  ]",
          "  name \"floor\"", "  boundingObject Box { size 2 2 0.01 }", "}"]

    # Add walls around the arena.
    half = sc.ARENA / 2
    for nm, (x, y, sx, sy) in {"wall_n": (0, half + 0.025, 2.1, 0.05), "wall_s": (0, -half - 0.025, 2.1, 0.05),
                               "wall_e": (half + 0.025, 0, 0.05, 2.1), "wall_w": (-half - 0.025, 0, 0.05, 2.1)}.items():
        L += ["Solid {", f"  translation {x} {y} 0.05", "  children [", "    Shape {",
              "      appearance PBRAppearance { baseColor 0.45 0.45 0.5 roughness 1 metalness 0 }",
              f"      geometry Box {{ size {sx} {sy} 0.1 }}", "    }", "  ]",
              f"  name \"{nm}\"", f"  boundingObject Box {{ size {sx} {sy} 0.1 }}", "}"]

    # Draw the sector markers. Hot sectors are orange.
    centres = sc.sector_centres()
    for s, (cx, cy) in enumerate(centres):
        hot = sc.SECTOR_WORK[s] > 5
        col = "0.95 0.65 0.25" if hot else ("0.78 0.88 0.80" if (s // sc.GRID + s % sc.GRID) % 2 == 0 else "0.70 0.82 0.74")
        L += ["Solid {", f"  translation {cx} {cy} 0.0005", "  children [", "    Shape {",
              f"      appearance PBRAppearance {{ baseColor {col} roughness 1 metalness 0 }}",
              "      geometry Box { size 0.47 0.47 0.001 }", "    }", "  ]", f"  name \"sector_{s}\"", "}"]

    # Add the obstacles to the arena.
    for i, (ox, oy, h) in enumerate(sc.OBSTACLES):
        L += ["Solid {", f"  translation {ox} {oy} 0.05", "  children [", "    Shape {",
              "      appearance PBRAppearance { baseColor 0.75 0.15 0.15 roughness 0.8 metalness 0 }",
              f"      geometry Box {{ size {2 * h} {2 * h} 0.1 }}", "    }", "  ]",
              f"  name \"obstacle_{i}\"", f"  boundingObject Box {{ size {2 * h} {2 * h} 0.1 }}", "}"]

    # Add the four e-puck robots and their communication devices.
    for i, (x, y, th) in enumerate(sc.STARTS):
        L += [f"DEF EPUCK{i} E-puck {{", f"  translation {x} {y} 0", f"  rotation 0 0 1 {th:.6f}",
              f"  name \"epuck{i}\"", "  controller \"swarm_agent\"", f"  controllerArgs [ \"{mode}\" ]",
              "  turretSlot [", "    GPS { name \"gps\" }", "    Compass { name \"compass\" }",
              "    Emitter { name \"swarm_tx\" }", "    Receiver { name \"swarm_rx\" }", "  ]", "}"]

    # Add the supervisor to record results and stop completed missions.
    L += ["Robot {", "  name \"supervisor\"", "  controller \"swarm_supervisor\"",
          f"  controllerArgs [ \"{mode}\" ]", "  supervisor TRUE", "}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webots", default="R2023b", help="Webots release name, e.g. R2023b or R2025a")
    args = ap.parse_args()
    out = os.path.join(HERE, "..", "worlds")
    os.makedirs(out, exist_ok=True)
    for mode, fname in (("distributed", "swarm_distributed.wbt"), ("centralised", "swarm_centralised.wbt"),
                        ("collect", "swarm_collect.wbt")):
        with open(os.path.join(out, fname), "w") as fh:
            fh.write(build_world(args.webots, mode))
        print("wrote", os.path.normpath(os.path.join(out, fname)), f"({args.webots})")


if __name__ == "__main__":
    main()
