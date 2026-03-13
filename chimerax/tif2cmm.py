import numpy as np
import tifffile


# 读取 TIFF 文件
def load_tif(file_path):
    return tifffile.imread(file_path)


# 将 TIFF 数据转换为 CMM 格式
def tif_to_cmm(tif_data, scale_factor=17.14):
    cmm_data = '<marker_set name="markers">\n'
    z, y, x = np.where(tif_data > 0)  # 获取所有大于 0 的点坐标
    # z, y, x = np.where(tif_data == 2)
    for idx, (zi, yi, xi) in enumerate(zip(z, y, x), start=1):
        # 将坐标乘以 scale_factor 并保留两位小数
        cmm_data += (
            f'<marker id="{idx}" x="{xi * scale_factor:.2f}" '
            f'y="{yi * scale_factor:.2f}" z="{zi * scale_factor:.2f}" '
            'r="1" g="1" b="0" radius="10"/>\n'
        )

    cmm_data += '</marker_set>\n'
    return cmm_data


# 保存 CMM 文件
def save_cmm(cmm_data, output_path):
    with open(output_path, 'w') as f:
        f.write(cmm_data)


# 主函数
def convert_tif_to_cmm(tif_path, cmm_output_path, scale_factor=17.14):
    tif_data = load_tif(tif_path)
    cmm_data = tif_to_cmm(tif_data, scale_factor)
    save_cmm(cmm_data, cmm_output_path)
    print(f"Converted CMM file saved to: {cmm_output_path}")


if __name__ == "__main__":
    # 示例使用
    tif_path = '/share/data/CryoET_Data/liushuo/dataset/IsoNet2/liucong/semantic map - _isonet2-n2n_unet-medium_TS_009_9.tif'  # 替换为实际的 TIFF 文件路径
    cmm_output_path = tif_path.replace('.tif', '.cmm')  # 替换为输出的 CMM 文件路径
    convert_tif_to_cmm(tif_path, cmm_output_path, scale_factor=1)
