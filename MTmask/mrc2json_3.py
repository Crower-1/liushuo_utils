import numpy as np
import json
from mrc.io import get_tomo
from scipy import ndimage
from skimage.feature import peak_local_max
import scipy.spatial.distance as dist
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull


def find_peak(semantic_mask):
    """
    Find peak points in the semantic mask using distance transform and local maxima detection.
    Args:
        semantic_mask (np.ndarray): 3D binary mask representing the microtubule segmentation.
    
    Returns:
        np.ndarray: Coordinates of local maxima.
    """
    distance_map = ndimage.distance_transform_edt(semantic_mask)
    local_maxi = peak_local_max(distance_map, min_distance=4, threshold_abs=0.2, exclude_border=False)
    return local_maxi

# 计算两点间的欧氏距离
def compute_distance_matrix(coordinates):
    return cdist(coordinates, coordinates)

# 计算两点的中点
def compute_midpoint(point1, point2):
    return (point1 + point2) / 2

# 检查中点是否在语义掩膜中
def check_midpoint_in_mask(midpoint, semantic_mask):
    z, y, x = midpoint.astype(int)
    if 0 <= z < semantic_mask.shape[0] and 0 <= y < semantic_mask.shape[1] and 0 <= x < semantic_mask.shape[2]:
        return semantic_mask[z, y, x] == 1
    return False

# 根据距离和可连线性生成连接矩阵
def generate_connectivity_matrix(coordinates, semantic_mask, distance_threshold=20):
    n = len(coordinates)
    connectivity_matrix = np.zeros((n, n), dtype=bool)
    
    distance_matrix = compute_distance_matrix(coordinates)
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < distance_threshold:
                midpoint = compute_midpoint(coordinates[i], coordinates[j])
                if check_midpoint_in_mask(midpoint, semantic_mask):
                    connectivity_matrix[i, j] = True
                    connectivity_matrix[j, i] = True
    return connectivity_matrix

# 计算向量夹角（返回度数）
def compute_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * (180 / np.pi)
    return angle

# 根据条件筛选符合要求的点
def filter_candidates(distance_matrix, connectivity_matrix, visited, current_point, distance_threshold=20):
    # 筛选条件：距离小于20像素，未访问且可以连线
    candidates_mask = (distance_matrix[current_point] < distance_threshold) & \
                      (connectivity_matrix[current_point]) & \
                      (~visited)
    return candidates_mask

# 划分点集
def divide_points_into_segments(coordinates, semantic_mask):
    n = len(coordinates)
    visited = np.zeros(n, dtype=bool)  # 访问记录
    distance_matrix = compute_distance_matrix(coordinates)
    connectivity_matrix = generate_connectivity_matrix(coordinates, semantic_mask)

    segments = []  # 存储所有的点集
    
    while np.sum(visited) < n:  # 直到所有点都被访问
        # 选择一个未被访问的点作为起始点
        unvisited_points = [i for i in range(n) if not visited[i]]
        if not unvisited_points:
            break
        start_point = unvisited_points[0]

        # 获取距离小于20像素、未访问且可以连线的最近点
        candidates_mask = filter_candidates(distance_matrix, connectivity_matrix, visited, start_point)
        candidates = np.where(candidates_mask)[0]

        if len(candidates) == 0:
            visited[start_point] = True
            continue  # 若没有符合条件的点，跳过当前点

        # 选择距离最近的点作为起始点集
        distances = distance_matrix[start_point, candidates]
        nearest_point = candidates[np.argmin(distances)]
        initial_segment = [start_point, nearest_point]
        visited[start_point] = True
        visited[nearest_point] = True

        # 扩展点集：从正方向开始
        def extend_segment(segment, direction=1):
            new_segment = segment.copy()
            extended = True
            while extended:
                extended = False
                current_point = new_segment[-1] if direction == 1 else new_segment[0]
                
                # 获取符合条件的点
                candidates_mask = filter_candidates(distance_matrix, connectivity_matrix, visited, current_point)
                candidates = np.where(candidates_mask)[0]
                
                if len(candidates) > 0:
                    distances = distance_matrix[current_point, candidates]
                    nearest_point = candidates[np.argmin(distances)]
                    
                    # 计算方向夹角
                    vector_current_to_new = coordinates[nearest_point] - coordinates[current_point]
                    vector_segment_end = coordinates[new_segment[-1]] - coordinates[new_segment[-2]] if len(new_segment) > 1 else np.zeros(3)
                    angle = compute_angle(vector_segment_end, vector_current_to_new)
                    
                    if angle < 30:
                        new_segment.append(nearest_point)
                        visited[nearest_point] = True
                        extended = True
            return new_segment
        
        # 从正方向延长
        forward_segment = extend_segment(initial_segment, direction=1)
        # 从负方向延长
        backward_segment = extend_segment(initial_segment, direction=-1)

        # 合并两个方向的点集
        segment = backward_segment[::-1] + forward_segment[1:]  # 逆序合并负方向和正方向的点集
        segments.append([coordinates[i] for i in segment])
    
    return segments

def process_microtubules(coordinates, semantic_mask, max_distance=30, angle_threshold=30, min_length=40):
    """
    Process the raw microtubule coordinates to separate, connect, and clean the microtubules.
    Args:
        coordinates (np.ndarray): Raw coordinates (z, y, x).
        semantic_mask (np.ndarray): 3D binary semantic mask representing the microtubules.
        max_distance (float): Maximum allowable distance between consecutive points.
        angle_threshold (float): Angle threshold for connecting points.
        min_length (float): Minimum length for a valid microtubule.
    
    Returns:
        list: List of cleaned and processed microtubules.
    """
    microtubules_segments = divide_points_into_segments(coordinates, semantic_mask)
    
    # segments = microtubules_segments
    
    # # Step 2: Connect disconnected microtubules
    # connected_segments = []
    # for segment in segments:
    #     if len(segment) < 2:
    #         continue  # Skip segments with only one point
        
    #     # Try to find the best connection with other segments
    #     for other_segment in segments:
    #         if segment != other_segment:
    #             # Compare the head and tail points of the two segments
    #             head = segment[0]
    #             tail = segment[-1]
    #             other_head = other_segment[0]
    #             other_tail = other_segment[-1]
                
    #             # Check if directions are approximately opposite
    #             vector1 = tail - head
    #             vector2 = other_tail - other_head
    #             if angle_between(vector1, -vector2) < 30 and np.linalg.norm(head - other_tail) < 40:
    #                 if is_valid_connection(head, other_tail, semantic_mask):
    #                     connected_segments.append(segment + other_segment)
                        
    # # Step 3: Delete small microtubules
    # final_segments = [segment for segment in connected_segments if np.linalg.norm(segment[-1] - segment[0]) >= min_length]

    return microtubules_segments

def actin_to_json(segments):
    """
    Convert the microtubule segments into a JSON-compatible format.
    Args:
        segments (list): List of microtubule segments, each containing a list of coordinates.
    
    Returns:
        str: JSON-formatted string of microtubule segments.
    """
    microtubules = []
    for idx, segment in enumerate(segments):
        microtubule = {
            "id": idx + 1,
            "points": segment.tolist(),
            "length": np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))  # Length of the segment
        }
        microtubules.append(microtubule)
    
    return json.dumps(microtubules, indent=4)

# Example usage
if __name__ == "__main__":
    # Load your semantic_mask (3D binary mask) here
    # semantic_mask = load_your_mask()
    
    semantic_mask = get_tomo('/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1033/MT/PO/pp1033_semantic_MT_label.mrc')

    # Find the peak points (center coordinates) from the mask
    centers = find_peak(semantic_mask)

    # Process the coordinates into segments
    processed_segments = process_microtubules(centers, semantic_mask)

    # Convert the processed segments to JSON format
    actin_json = actin_to_json(processed_segments)

    # Optionally, write the result to a file
    with open('/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1033/MT/PO/actin_segments.json', 'w') as f:
        f.write(actin_json)
