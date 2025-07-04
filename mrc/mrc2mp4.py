import mrcfile
import numpy as np
import cv2

# 读取MRC文件
mrc_file_path = '/home/liushuo/Documents/data/stack-out_demo/pp1176/ves_seg/pp1176_wbp_corrected.mrc'
with mrcfile.open(mrc_file_path, permissive=True) as mrc:
    data = mrc.data

# 获取数据的维度
num_slices, height, width = data.shape

# 在三维数据上进行归一化
data_normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
data_normalized = data_normalized.astype(np.uint8)

# 设置视频参数
output_video_path = './pp1176.mp4'
fps = 60  # 每秒帧数，可以根据需要调整
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 创建视频写入对象
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=False)

# 遍历3D数据的每一层，并写入视频
for i in range(num_slices):
    frame = data_normalized[i]
    out.write(frame)

# 释放视频写入对象
out.release()

print(f'视频已保存到 {output_video_path}')