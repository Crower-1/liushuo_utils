import mrcfile
import numpy as np

# 读取 MRC 文件
def load_mrc(file_path):
    with mrcfile.open(file_path, 'r') as mrc:
        return mrc.data

# 将 MRC 数据转换为 CMM 格式
def mrc_to_cmm(mrc_data, scale_factor=17.14):
    cmm_data = '<marker_set name="markers">\n'
    z, y, x = np.where(mrc_data > 0)  # 获取所有大于 0 的点坐标
    # z, y, x = np.where(mrc_data == 2)
    for idx, (zi, yi, xi) in enumerate(zip(z, y, x), start=1):
        # 将坐标乘以 scale_factor 并保留两位小数
        cmm_data += f'<marker id="{idx}" x="{xi * scale_factor:.2f}" y="{yi * scale_factor:.2f}" z="{zi * scale_factor:.2f}" r="1" g="1" b="0" radius="10"/>\n'

    cmm_data += '</marker_set>\n'
    return cmm_data

# 保存 CMM 文件
def save_cmm(cmm_data, output_path):
    with open(output_path, 'w') as f:
        f.write(cmm_data)

# 主函数
def convert_mrc_to_cmm(mrc_path, cmm_output_path, scale_factor=17.14):
    mrc_data = load_mrc(mrc_path)
    cmm_data = mrc_to_cmm(mrc_data, scale_factor)
    save_cmm(cmm_data, cmm_output_path)
    print(f"Converted CMM file saved to: {cmm_output_path}")
    
# if __name__ == "__main__":
#     # # base_names = ['pp0039', 'pp123', 'pp132', 'pp142', 'pp0366', 'pp0402', 'pp0680']
#     # base_names = ['pp142']
#     # # 示例使用
#     # for base_name in base_names:
#     #     mrc_path = f'/media/liushuo/data1/data/fig_demo/{base_name}/membrane/{base_name}_membrane_label.mrc'  # 替换为实际的 MRC 文件路径
#     #     cmm_output_path = f'/media/liushuo/data1/data/fig_demo/{base_name}/membrane/keypoint.cmm'  # 替换为输出的 CMM 文件路径

#     #     convert_mrc_to_cmm(mrc_path, cmm_output_path)
    
    # # 示例使用
mrc_path = '/media/liushuo/data2/data/xyn_demo/type2.mrc'  # 替换为实际的 MRC 文件路径
cmm_output_path = mrc_path.replace('.mrc', '.cmm')  # 替换为输出的 CMM 文件路径
# cmm_output_path = '/media/liushuo/data1/data/fig_demo_2/p255/membrane/keypoints.cmm'  # 替换为输出的 CMM 文件路径

convert_mrc_to_cmm(mrc_path, cmm_output_path, scale_factor=18.12)



