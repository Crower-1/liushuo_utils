import os
import torch
import SimpleITK as sitk
import numpy as np
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from mrc.io import get_tomo, save_tomo
import torch, inspect

REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = "/home/liushuo/isensee/temp"

# download_path = snapshot_download(
#     repo_id=REPO_ID,
#     allow_patterns=[f"{MODEL_NAME}/*"],
#     local_dir=DOWNLOAD_DIR
# )

# 原始坐标文件中的像素尺寸（nm）
ORIG_PIXEL_SIZE = 17.14
# 目标图像的像素尺寸（nm）
TARGET_PIXEL_SIZE = 17.14
# 缩放比例：coord 中的坐标 × SCALE → 在 target 图像中的索引
SCALE = ORIG_PIXEL_SIZE / TARGET_PIXEL_SIZE

tomo_path = '/media/liushuo/data2/data/demo_img/2025_05_13/pace03_ts_005/pace03_ts_005.mrc'
coords_path = '/share/data/CryoET_Data/xiayn/local/work-now/demo/chimera_demo/rec_pace03_ts_005_all/pace03_ts_005.coords'

# --- 辅助函数 ---
def get_points_from_coords(path):
    """
    读取 coords 文件，提取每行的 x,y,z 三个浮点数，按 SCALE 缩放并四舍五入取整。
    返回的列表格式为 [(x1, y1, z1), (x2, y2, z2), ...]。
    """
    points = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # 最后三个是 x, y, z
            x_ori, y_ori, z_ori = map(float, parts[-3:])
            # 缩放并四舍五入
            x = int(round(x_ori * SCALE))
            y = int(round(y_ori * SCALE))
            z = int(round(z_ori * SCALE))
            points.append((x, y, z))
    return points

# --- 下载并初始化模型 ---
model_folder = os.path.join(DOWNLOAD_DIR, MODEL_NAME)

session = nnInteractiveInferenceSession(
    device=torch.device("cuda:0"),
    use_torch_compile=False,
    verbose=False,
    torch_n_threads=os.cpu_count(),
    do_autozoom=True,
    use_pinned_memory=True,
)
session.initialize_from_trained_model_folder(model_folder)

# def describe_module(name, mod):
#     print(f"\n=== {name} ===")
#     print("type:", type(mod))
#     try:
#         n_params = sum(p.numel() for p in mod.parameters())
#         devs = sorted({str(p.device) for p in mod.parameters()})
#         dtypes = sorted({str(p.dtype) for p in mod.parameters()})
#         print(f"params: {n_params:,}")
#         print("devices:", devs)
#         print("dtypes:", dtypes)
#         # 部分 state_dict 键名
#         sd_keys = list(mod.state_dict().keys())
#         print("state_dict keys (first 15):", sd_keys[:15])
#     except Exception as e:
#         print("could not inspect as nn.Module:", e)

#     # TorchScript 的额外信息
#     if isinstance(mod, torch.jit.ScriptModule) or isinstance(mod, torch.jit.RecursiveScriptModule):
#         try:
#             print("\nTorchScript code:\n", mod.code[:800], "...\n")
#         except Exception:
#             pass

# net = session.network
# save_dir = '/home/liushuo/isensee/temp/nnInteractive_v1.0/fold_0/'
# enc_sd = net.encoder.state_dict()
# dec_sd = net.decoder.state_dict()
# torch.save(enc_sd, f"{save_dir}/encoder_only.pth")
# torch.save(dec_sd, f"{save_dir}/decoder_only.pth")
# print(session.network)                 # 全部结构树
# print("\n-- encoder --\n", session.network.encoder)
# print("\n-- decoder --\n", session.network.decoder)
# print("\n-- seg_outputs (heads) --\n", session.network.seg_outputs)
# print("conv_op:", session.network.conv_op)  # 通常是 nn.Conv3d

# # 1) 粗暴枚举 session 里名字包含 net/model 的字段
# candidates = {}
# for k, v in vars(session).items():
#     if any(t in k.lower() for t in ["net", "model", "encoder", "decoder"]):
#         candidates[k] = v

# # 2) 优先处理常见结构：dict/list/单个模块
# for k, v in list(candidates.items()):
#     if isinstance(v, dict):
#         for subk, subv in v.items():
#             if hasattr(subv, "parameters") or isinstance(subv, (torch.nn.Module, torch.jit.ScriptModule)):
#                 describe_module(f"{k}['{subk}']", subv)
#     elif isinstance(v, (list, tuple)):
#         for i, subv in enumerate(v):
#             if hasattr(subv, "parameters") or isinstance(subv, (torch.nn.Module, torch.jit.ScriptModule)):
#                 describe_module(f"{k}[{i}]", subv)
#     else:
#         if hasattr(v, "parameters") or isinstance(v, (torch.nn.Module, torch.jit.ScriptModule)):
#             describe_module(k, v)

# # 3) 兜底：看看类定义位置，方便你直接打开源码
# import nnInteractive.inference.inference_session as infmod
# print("\n----- nnInteractiveInferenceSession is defined in:", inspect.getfile(infmod))
# print("Class source available:", hasattr(infmod, "nnInteractiveInferenceSession"))

# --- 读取图像并做维度检查 ---
input_itk = get_tomo(tomo_path)  # 假设返回 SimpleITK Image 类型
# img_np = sitk.GetArrayFromImage(input_itk)  # shape: (z, y, x)
# 转成 (1, x, y, z) 或者 (1, z, y, x) 视 nnInteractive 要求而定
# 下面举例保持原脚本风格，假设需要 (1, x, y, z)
img = input_itk.transpose(2, 1, 0)[None]  

if img.ndim != 4:
    raise ValueError("Input image must be 4D with shape (1, x, y, z)")

# --- 预分配结果缓冲，单通道二值 0/1 ---
results = np.zeros_like(img, dtype=np.uint8)  # shape (1, x, y, z)

# --- 读取并缩放点坐标 ---
points = get_points_from_coords(coords_path)
if not points:
    raise RuntimeError("No valid points found in coords file.")

# --- 一次性将图像设置进 session ---
session.set_image(img)

# 针对每个点进行分割，并赋予唯一标签
for label_idx, pt in enumerate(points, start=1):
    # 清零新的 target buffer
    target = torch.zeros(img.shape[1:], dtype=torch.uint8, device=session.device)
    session.set_target_buffer(target)

    # 添加正交互点
    session.add_point_interaction(pt, include_interaction=True)

    # 取出 mask 并二值化
    mask = session.target_buffer.cpu().numpy()
    mask = (mask > 0).astype(bool)  # 布尔索引稍微高效

    # 只给还没标记的位置赋当前标签
    # results[0] 形状为 (x, y, z)
    new_region = mask & (results[0] == 0)
    results[0][new_region] = label_idx

    # 重置交互，为下一个点做准备
    session.reset_interactions()

# --- 保存最终拼接结果 ---

results = results[0].transpose(2, 1, 0)  # 转回 (z, y, x) 

# 构造输出目录：在 tomo_path 同目录下新建 ribo 文件夹
input_dir = os.path.dirname(tomo_path)
ribo_dir  = os.path.join(input_dir, "ribo")
os.makedirs(ribo_dir, exist_ok=True)

# 构造输出文件名：将原名 xx_wbp_corrected.mrc 改为 xx_ribo_volume.mrc
base  = os.path.basename(tomo_path)                   # e.g. "pp267_wbp_corrected.mrc"
stem  = os.path.splitext(base)[0]                    # e.g. "pp267_wbp_corrected"
suffix = "_wbp_corrected"
if stem.endswith(suffix):
    prefix = stem[:-len(suffix)]
else:
    prefix = stem
out_name = f"{prefix}_ribo_volume.mrc"                # e.g. "pp267_ribo_volume.mrc"
out_path = os.path.join(ribo_dir, out_name)
save_tomo(results, out_path, voxel_size=TARGET_PIXEL_SIZE, datetype=np.uint16)

print(f"Saved binary result: {out_path}")