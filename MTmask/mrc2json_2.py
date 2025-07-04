import numpy as np
import json
from mrc.io import get_tomo
from scipy import ndimage
from skimage.feature import peak_local_max
import scipy.spatial.distance as dist


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

def angle_between(v1, v2):
    """
    Calculate the angle between two vectors.
    Args:
        v1, v2 (np.ndarray): Input vectors.
    
    Returns:
        float: Angle in degrees between the two vectors.
    """
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi
    return angle

def is_valid_connection(start, end, mask, max_distance=30):
    """
    Check if two points are valid for connecting.
    Args:
        start, end (np.ndarray): Coordinates of the points.
        mask (np.ndarray): The semantic mask (3D).
        max_distance (float): Maximum allowable distance between points for connection.
    
    Returns:
        bool: True if the connection is valid, False otherwise.
    """
    # Check if the distance between the points is within the limit
    dist_between = np.linalg.norm(start - end)
    if dist_between > max_distance:
        return False
    
    # Check if the points between are non-zero in the mask
    path = np.linspace(start, end, num=10)  # Create 10 intermediate points between start and end
    for point in path.astype(int):
        if mask[tuple(point)] == 0:
            return False
    return True

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
    # Step 1: Separate microtubules by checking distances and angles
    segments = []
    visited = np.zeros(len(coordinates), dtype=bool)
    
    def find_next_point(start_idx, direction=1):
        """Find the next valid point in a given direction."""
        nonlocal visited
        current_point = coordinates[start_idx]
        current_segment = [current_point]
        visited[start_idx] = True
        
        # Continue until we can no longer find valid points
        idx = start_idx + direction
        while 0 <= idx < len(coordinates) and not visited[idx]:
            next_point = coordinates[idx]
            # Check angle between current and next point
            angle = angle_between(current_point - coordinates[start_idx-1] if direction == -1 else coordinates[start_idx+1] - current_point, next_point - current_point)
            if angle < angle_threshold and np.linalg.norm(current_point - next_point) < max_distance:
                current_segment.append(next_point)
                visited[idx] = True
                current_point = next_point
            else:
                break
            idx += direction
        
        return current_segment
    
    for idx in range(len(coordinates)):
        if not visited[idx]:
            # Create forward and backward segments for each point
            forward_segment = find_next_point(idx, direction=1)
            backward_segment = find_next_point(idx, direction=-1)
            segments.append(forward_segment + backward_segment)
    
    # Step 2: Connect disconnected microtubules
    connected_segments = []
    for segment in segments:
        if len(segment) < 2:
            continue  # Skip segments with only one point
        
        # Try to find the best connection with other segments
        for other_segment in segments:
            if segment != other_segment:
                # Compare the head and tail points of the two segments
                head = segment[0]
                tail = segment[-1]
                other_head = other_segment[0]
                other_tail = other_segment[-1]
                
                # Check if directions are approximately opposite
                vector1 = tail - head
                vector2 = other_tail - other_head
                if angle_between(vector1, -vector2) < 30 and np.linalg.norm(head - other_tail) < 40:
                    if is_valid_connection(head, other_tail, semantic_mask):
                        connected_segments.append(segment + other_segment)
                        
    # Step 3: Delete small microtubules
    final_segments = [segment for segment in connected_segments if np.linalg.norm(segment[-1] - segment[0]) >= min_length]

    return final_segments

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
    
    semantic_mask = get_tomo('/media/liushuo/data1/data/synapse_seg/pp463/Prediction/MT.mrc')

    # Find the peak points (center coordinates) from the mask
    centers = find_peak(semantic_mask)

    # Process the coordinates into segments
    processed_segments = process_microtubules(centers, semantic_mask)

    # Convert the processed segments to JSON format
    actin_json = actin_to_json(processed_segments)

    # Optionally, write the result to a file
    with open('/media/liushuo/data1/data/synapse_seg/pp463/Prediction/MT.json', 'w') as f:
        f.write(actin_json)
