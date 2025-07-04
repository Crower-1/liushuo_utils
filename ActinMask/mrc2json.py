import os
import json
import numpy as np
import mrcfile as mf
from skimage.measure import label
from skimage.morphology import skeletonize_3d
from scipy import ndimage
from scipy.spatial import distance
from itertools import combinations

class ActinPostProcessor:
    def __init__(self, radius=2):
        self.radius = radius

    def read_mrc(self, mrc_path):
        """读取MRC文件并返回数据数组"""
        with mf.open(mrc_path, permissive=True) as mrc:
            data = mrc.data.copy()
            voxel_size = mrc.voxel_size
        return data, voxel_size

    def label_instances(self, semantic_mask):
        """对语义分割结果进行3D连通组件标记"""
        labeled_mask, num_features = label(semantic_mask, return_num=True, connectivity=1)
        print(f"找到 {num_features} 个actin实例。")
        return labeled_mask, num_features

    def extract_skeleton(self, semantic_mask):
        """对所有实例的语义掩码进行骨架化"""
        skeleton = skeletonize_3d(semantic_mask)
        return skeleton

    def find_endpoints(self, skeleton):
        """
        找到骨架的所有端点。
        端点定义为邻居数为1的点。
        """
        struct = np.ones((3,3,3), dtype=np.int32)
        neighbors = ndimage.convolve(skeleton.astype(int), struct, mode='constant', cval=0)
        neighbors = neighbors * skeleton
        # 端点：邻居数为2（包括自身）
        endpoints = np.where((skeleton) & (neighbors == 2))
        endpoint_coords = list(zip(endpoints[0], endpoints[1], endpoints[2]))
        return endpoint_coords

    def get_neighbor_points(self, point, skeleton, min_dist=5, max_dist=10):
        """
        获取距离当前点在[min_dist, max_dist]之间的最近骨架点
        """
        z, y, x = point
        z_min, z_max = max(z - max_dist, 0), min(z + max_dist +1, skeleton.shape[0])
        y_min, y_max = max(y - max_dist, 0), min(y + max_dist +1, skeleton.shape[1])
        x_min, x_max = max(x - max_dist, 0), min(x + max_dist +1, skeleton.shape[2])

        sub_skeleton = skeleton[z_min:z_max, y_min:y_max, x_min:x_max]
        if not np.any(sub_skeleton):
            return []

        # 获取子区域的坐标
        sub_indices = np.array(np.nonzero(sub_skeleton)).T + np.array([z_min, y_min, x_min])

        # 计算距离
        distances = np.linalg.norm(sub_indices - np.array(point), axis=1)
        mask = (distances > min_dist) & (distances < max_dist)
        valid_points = sub_indices[mask].tolist()
        return valid_points

    def calculate_angle(self, p1, p2, p3):
        """
        计算向量p1->p2和p2->p3之间的夹角
        """
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        # 防止数值问题
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    def extract_keypoints(self, instance_skeleton, endpoints, max_angle=20):
        """
        根据给定的骨架和端点提取关键点列表。
        支持多个关键点列表，每个列表对应一个actin。
        """
        if len(endpoints) < 2:
            return []  # 跳过只有一个端点的情况

        keypoints_lists = []
        visited = set()

        def traverse(start_point, other_endpoints):
            path = [start_point]
            visited.add(tuple(start_point))
            current_point = start_point
            previous_point = None

            while True:
                neighbors = self.get_neighbor_points(current_point, instance_skeleton)
                # 排除已访问的点
                neighbors = [pt for pt in neighbors if tuple(pt) not in visited]
                if not neighbors:
                    break

                # 寻找距离合适的最近点
                neighbors = sorted(neighbors, key=lambda p: np.linalg.norm(np.array(p) - np.array(current_point)))
                next_point = None

                for pt in neighbors:
                    if previous_point is None:
                        next_point = pt
                        break
                    angle = self.calculate_angle(previous_point, current_point, pt)
                    if angle <= max_angle:
                        next_point = pt
                        break

                if next_point is None:
                    break

                # 检查是否接近其他端点
                close_to_other = False
                for oe in other_endpoints:
                    if np.linalg.norm(np.array(next_point) - np.array(oe)) <= 5:
                        close_to_other = True
                        break
                if close_to_other:
                    path.append(next_point)
                    visited.add(tuple(next_point))
                    break

                path.append(next_point)
                visited.add(tuple(next_point))
                previous_point = current_point
                current_point = next_point

            return path

        # 如果只有两个端点
        if len(endpoints) == 2:
            path = traverse(endpoints[0], [endpoints[1]])
            if len(path) > 1:
                keypoints_lists.append(path)
        else:
            # 多个端点的情况
            for ep in endpoints:
                if tuple(ep) in visited:
                    continue
                path = traverse(ep, [e for e in endpoints if e != ep])
                if len(path) > 1:
                    keypoints_lists.append(path)

        return keypoints_lists

    def process(self, semantic_mrc_path, output_json_path):
        """主处理函数"""
        # 1. 读取语义分割MRC文件
        semantic_mask, voxel_size = self.read_mrc(semantic_mrc_path)
        print(f"MRC数据维度: {semantic_mask.shape}, voxel size: {voxel_size}")

        # 2. 实例分割
        labeled_mask, num_features = self.label_instances(semantic_mask)

        # 3. 提取所有实例的骨架
        skeleton = self.extract_skeleton(semantic_mask)
        if not np.any(skeleton):
            print("骨架提取失败，所有实例跳过。")
            return

        # 4. 遍历每个实例，提取关键点和长度
        actin_list = []
        actin_id_counter = 1  # 唯一actin_id计数器

        for actin_id in range(1, num_features + 1):
            print(f"处理actin实例ID: {actin_id}")
            instance_mask = (labeled_mask == actin_id)
            if np.sum(instance_mask) == 0:
                continue  # 跳过空实例

            # 4.1 获取实例的骨架
            instance_skeleton = np.logical_and(instance_mask, skeleton)
            if not np.any(instance_skeleton):
                print(f"实例ID {actin_id} 的骨架提取失败，跳过。")
                continue

            # 4.2 找到骨架端点
            endpoints = self.find_endpoints(instance_skeleton)
            if len(endpoints) < 2:
                print(f"实例ID {actin_id} 端点数量少于2，跳过。")
                continue

            # 4.3 提取关键点列表
            keypoints_lists = self.extract_keypoints(instance_skeleton, endpoints)

            if not keypoints_lists:
                print(f"实例ID {actin_id} 没有找到关键点列表，跳过。")
                continue

            # 4.4 为每个关键点列表分配唯一的actin_id，并计算长度
            for kp_list in keypoints_lists:
                if len(kp_list) < 2:
                    continue  # 跳过长度不足的关键点列表

                length = self.calculate_length(kp_list)

                actin_dict = {
                    "id": actin_id_counter,
                    # "original_instance_id": int(actin_id),
                    "points": [[float(z), float(y), float(x)] for z, y, x in kp_list],
                    "length": float(length)
                }
                actin_list.append(actin_dict)
                actin_id_counter += 1

        # 5. 保存到JSON文件
        with open(output_json_path, "w") as json_file:
            json.dump(actin_list, json_file, indent=4)
        print(f"关键点JSON文件已保存到: {output_json_path}")

    def calculate_length(self, keypoints):
        """根据关键点计算actin的长度"""
        if len(keypoints) < 2:
            return 0.0
        # 使用向量化计算连续关键点之间的欧氏距离
        keypoints_array = np.array(keypoints)
        deltas = np.diff(keypoints_array, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        length = np.sum(distances)
        return length



if __name__ == "__main__":
    # import argparse
    from scipy import ndimage

    # parser = argparse.ArgumentParser(description="将actin语义分割转换为实例分割并提取关键点。")
    # parser.add_argument("semantic_mrc_path", type=str, help="actin语义分割结果的MRC文件路径。")
    # parser.add_argument("output_json_path", type=str, help="输出的关键点JSON文件路径。")
    # args = parser.parse_args()
    semantic_mrc_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1033/actin/pp1033_semantic_actin_label.mrc'
    output_json_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1033/actin/pp1033_actin_point.json'

    processor = ActinPostProcessor(radius=2)
    processor.process(semantic_mrc_path, output_json_path)
