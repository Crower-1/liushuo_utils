import mrcfile
import numpy as np

def invert_and_save_mrc(input_file, output_file, voxel_size=17.14):
    # 读取 mrc 文件
    with mrcfile.open(input_file, mode='r') as mrc:
        
        # 获取原始图像数据
        image_data = -mrc.data.astype(np.float32)
        voxel_size = mrc.voxel_size
      
    
    int8_image = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 127

    # 保存反转后的图像到新的 mrc 文件
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(int8_image.astype(np.int8))
        mrc.voxel_size = voxel_size

    print(f"反转后的图像已保存到 {output_file}")
    
def invert_and_save_mrc_float32(input_file, output_file, voxel_size=17.14):
    # 读取 mrc 文件
    with mrcfile.open(input_file, mode='r') as mrc:
        
        # 获取原始图像数据
        image_data = -mrc.data.astype(np.float32)

    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    # 保存反转后的图像到新的 mrc 文件
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(image_data)
        mrc.voxel_size = voxel_size

    print(f"反转后的图像已保存到 {output_file}")


# base_name = 'pp0680'
# # 使用示例
# input_file = f'/media/liushuo/data1/data/tcl_demo/{base_name}/ves_seg/{base_name}_wbp_corrected.mrc'  # 替换为实际输入文件路径
# output_file = f'/media/liushuo/data1/data/tcl_demo/{base_name}/{base_name}_reverse_uint8.mrc'  # 替换为实际输出文件路径

# base_name = 'p287'
input_file = f'/share/data/CryoET_Data/liushuo/dataset/IsoNet2/liucong_in_tissue/corrected_tomos/_isonet2-n2n_unet-medium_TS_173_10.96Apx.mrc'  # 替换为实际输入文件路径
# output_file = f'/media/liushuo/data1/data/fig_demo_2/pp235/pp235-bin4-5i_reverse.mrc'  # 替换为实际输出文件路径
output_file = input_file.replace('.mrc', '_reverse.mrc')  # 替换为输出的 MRC 文件路径

invert_and_save_mrc(input_file, output_file)

# base_names = ['p90', 'p218', 'p260']
# working_dir = '/media/liushuo/data1/data/fig_demo_2/'
# for base_name in base_names:
#     input_file = f'{working_dir}/{base_name}/{base_name}.mrc'  # 替换为实际输入文件路径
#     output_file = f'{working_dir}/{base_name}/{base_name}_reverse.mrc'  # 替换为实际输出文件路径
#     invert_and_save_mrc(input_file, output_file)