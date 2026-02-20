import numpy as np
import trimesh

def compute_scene_center_from_ply(ply_path: str):
    """
    Load PLY point cloud and compute a robust scene center using median.
    """
    mesh = trimesh.load(ply_path)
    points = mesh.vertices  # (N, 3)

    # median is robust to outliers (background points)
    scene_center = np.median(points, axis=0)

    return scene_center

if __name__ == "__main__":
    ply_path = "/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
    scene_center = compute_scene_center_from_ply(ply_path)
    print(scene_center) ## [1.72378379 2.05881643 7.34031439]