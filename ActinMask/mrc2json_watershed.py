import os
import json
import numpy as np
import mrcfile as mf
from skimage.measure import label
from skimage.morphology import skeletonize_3d
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

class ActinPostProcessor:
    def __init__(self, radius=2, keypoint_interval=15):
        """
        初始化处理器，设定参数：
        - radius: Actin 细丝的半径
        - keypoint_interval: 关键点间隔（单位：像素）
        """
        self.radius = radius
        self.keypoint_interval = keypoint_interval

    def read_mrc(self, mrc_path):
        """读取 MRC 文件并返回数据数组"""
        with mf.open(mrc_path, permissive=True) as mrc:
            data = mrc.data.copy()
            voxel_size = mrc.voxel_size
        return data, voxel_size

    def separate_connected_actin(self, semantic_mask):
        """
        分离连接的 Actin 实例，使用距离变换和 Watershed 分水岭算法
        """
        # 计算距离变换
        distance_map = ndimage.distance_transform_edt(semantic_mask)
        
        # 找到局部最大值（新版本peak_local_max没有indices参数，返回的是坐标）
        local_maxi = peak_local_max(distance_map, min_distance=4, threshold_abs=0.2, exclude_border=False)
        
        # 使用local_maxi生成markers，并确保与semantic_mask大小匹配
        markers = np.zeros_like(semantic_mask, dtype=int)
        for i, peak in enumerate(local_maxi):
            markers[tuple(peak)] = i + 1  # 标记每个局部最大值为唯一的实例ID
        
        # 使用Watershed算法进行分水岭分割
        labeled_mask = watershed(-distance_map, markers, mask=semantic_mask)
        return labeled_mask

    def extract_keypoints(self, skeleton):
        """
        提取骨架上的关键点，间隔为self.keypoint_interval
        """
        # 计算骨架每个点的邻域
        struct = np.ones((3, 3, 3), dtype=int)
        distance_transform = ndimage.distance_transform_edt(skeleton)
        
        keypoints = []
        for z in range(skeleton.shape[0]):
            for y in range(skeleton.shape[1]):
                for x in range(skeleton.shape[2]):
                    if skeleton[z, y, x]:
                        # 获取当前点到骨架末端的距离
                        if distance_transform[z, y, x] % self.keypoint_interval == 0:
                            keypoints.append([z, y, x])
        
        return keypoints

    def process(self, semantic_mrc_path, output_json_path):
        """
        处理 MRC 文件，提取实例和关键点，并保存为 JSON
        """
        # 读取 MRC 文件
        semantic_mask, _ = self.read_mrc(semantic_mrc_path)
        
        # 分离连接的 Actin 实例
        labeled_mask = self.separate_connected_actin(semantic_mask)
        
        # 对每个实例提取骨架
        skeleton = skeletonize_3d(labeled_mask > 0)
        
        # 提取关键点
        keypoints = self.extract_keypoints(skeleton)
        
        # 将结果保存到 JSON 文件
        result = {
            'instances': []
        }
        
        # 假设每个实例的 ID 是其在 labeled_mask 中的唯一标记
        for instance_id in np.unique(labeled_mask):
            if instance_id == 0:
                continue  # 忽略背景
            instance_mask = labeled_mask == instance_id
            skeleton = skeletonize_3d(instance_mask)
            instance_keypoints = self.extract_keypoints(skeleton)
            
            result['instances'].append({
                'id': int(instance_id),
                'keypoints': instance_keypoints
            })
        
        # 保存 JSON
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"结果已保存到 {output_json_path}")

if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="将actin语义分割转换为实例分割并提取关键点。")
    # parser.add_argument("semantic_mrc_path", type=str, help="actin语义分割结果的MRC文件路径。")
    # parser.add_argument("output_json_path", type=str, help="输出的关键点JSON文件路径。")
    # args = parser.parse_args()

    # processor = ActinPostProcessor(radius=2, keypoint_interval=15)
    # processor.process(args.semantic_mrc_path, args.output_json_path)
    
    semantic_mrc_path = '/media/liushuo/data1/data/synapse_seg/pp463/Prediction/actin.mrc'
    output_json_path = '/media/liushuo/data1/data/synapse_seg/pp463/Prediction/actin/points.json'

    processor = ActinPostProcessor(radius=2, keypoint_interval=15)
    processor.process(semantic_mrc_path, output_json_path)
