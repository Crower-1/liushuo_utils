import mrcfile
import numpy as np
import h5py

def hdf_to_mrc(hdf_filename, mrc_filename):
    """
    将 HDF 文件中 '/data' 数据集转换为 MRC 文件

    参数：
        hdf_filename: 输入 HDF 文件路径
        mrc_filename: 输出 MRC 文件路径
    """
    # 打开 HDF 文件，读取 '/data' 数据集
    with h5py.File(hdf_filename, 'r') as hdf:
        # if '/images' not in hdf:
        #     raise ValueError(f"HDF 文件 {hdf_filename} 中不存在 '/data' 数据集")
        # data = hdf['MDF/images/0/image'][:]
        data = hdf['mito_pred'][:]
        print(f"读取 HDF 文件 {hdf_filename} 中的 '/data' 数据集，数据形状：{data.shape}")

    # 将数据写入 MRC 文件
    with mrcfile.new(mrc_filename, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        # mrc.update_header_from_data()
        print(f"数据已保存为 MRC 文件：{mrc_filename}")

def mrc_to_hdf(mrc_filename, hdf_filename):
    """
    将 MRC 文件转换为 HDF 文件，并保存为 '/data' 数据集

    参数：
        mrc_filename: 输入 MRC 文件路径
        hdf_filename: 输出 HDF 文件路径
    """
    # 打开 MRC 文件，读取数据
    with mrcfile.open(mrc_filename, permissive=True) as mrc:
        data = mrc.data
        print(f"读取 MRC 文件 {mrc_filename} 中的数据，数据形状：{data.shape}")

    # 将数据写入 HDF 文件，并保存到 '/data' 数据集
    with h5py.File(hdf_filename, 'w') as hdf:
        hdf.create_dataset('/data', data=data)
        print(f"数据已保存为 HDF 文件：{hdf_filename}，数据集为 '/data'")

# 示例用法
mrc_file_path = "/home/liushuo/Downloads/BACHD_bin4_output.mrc"
hdf_file_path = "/home/liushuo/Downloads/mito-seg.hdf"
hdf_to_mrc(hdf_file_path, mrc_file_path)
