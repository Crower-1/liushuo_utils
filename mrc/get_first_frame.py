from mrc.io import get_tomo, save_tomo
import numpy

all_frame_data = get_tomo("/home/liushuo/Documents/data/input.mrc")
first_frame_data = all_frame_data[0:19]
save_tomo(first_frame_data, "/home/liushuo/Documents/data/output.mrc")