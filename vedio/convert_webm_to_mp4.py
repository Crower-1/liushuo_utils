#!/usr/bin/env python3
import subprocess
import sys
import os

def convert_webm_to_h264(input_path: str, output_path: str,
                         video_bitrate: str = '500k',
                         audio_bitrate: str = '128k',
                         preset: str = 'medium',
                         crf: int = 23) -> None:
    """
    将 WebM 视频转为 H.264 编码的 MP4 视频。

    参数:
        input_path:  输入 .webm 文件路径
        output_path: 输出 .mp4 文件路径（建议以 .mp4 结尾）
        video_bitrate:  目标视频码率，如 '2M'、'500k'
        audio_bitrate:  目标音频码率，如 '128k'
        preset:        FFmpeg 的压缩预设（ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow）
        crf:           常量质量系数，范围 0 (无损)–51 (最差)，常用 18–28 范围
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    # 构造 FFmpeg 命令
    cmd = [
        'ffmpeg',
        '-y',                    # 覆盖已存在的输出文件
        '-i', input_path,        # 输入文件
        '-c:v', 'libx264',       # 视频编码器
        '-preset', preset,       # 压缩预设
        '-crf', str(crf),        # 质量系数
        '-b:v', video_bitrate,   # 目标视频码率
        '-c:a', 'aac',           # 音频编码器
        '-b:a', audio_bitrate,   # 目标音频码率
        output_path
    ]

    # 调用 FFmpeg
    try:
        subprocess.run(cmd, check=True)
        print(f"转换完成：{output_path}")
    except subprocess.CalledProcessError as e:
        print("转码失败，FFmpeg 返回码：", e.returncode)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python convert.py 输入.webm 输出.mp4")
        sys.exit(0)

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    convert_webm_to_h264(in_file, out_file)