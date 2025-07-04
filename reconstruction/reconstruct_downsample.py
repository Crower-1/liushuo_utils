import re
import numpy as np
import mrcfile
import argparse
import os
import time
from scipy import fft
from tqdm import tqdm
import numba

class TomoReconstructor:
    """
    基础层析重建类，定义了加载数据、保存结果、下采样及共用的加权方法
    """
    def __init__(self, input_file, output_file, tilt_file, vol_z=None):
        self.input_file = input_file
        self.output_file = output_file
        self.tilt_file = tilt_file
        self.vol_z = vol_z
        self.tilt_series = None
        self.tilt_angles = np.loadtxt(self.tilt_file)
        self.pixel_size = 1.0
        self.cos_sin_values = None
        self.volume = None
        self.proj_x = None
        self.proj_y = None

    def load_data(self):
        print(f"Loading tilt series from {self.input_file}")
        with mrcfile.open(self.input_file) as mrc:
            self.tilt_series = mrc.data.astype(np.float32)
            if hasattr(mrc, 'voxel_size'):
                self.pixel_size = mrc.voxel_size.x

        n_tilts, self.proj_y, self.proj_x = self.tilt_series.shape
        if self.vol_z is None:
            self.vol_z = self.proj_y

        print(f"Tilt series dimensions: {n_tilts} projections of {self.proj_y} x {self.proj_x}")
        print(f"Output volume size: {self.proj_x} x {self.proj_y} x {self.vol_z}")

        self.cos_sin_values = np.zeros((len(self.tilt_angles), 2), dtype=np.float32)
        for i, angle in enumerate(self.tilt_angles):
            angle_rad = np.deg2rad(angle)
            self.cos_sin_values[i, 0] = np.cos(angle_rad)
            self.cos_sin_values[i, 1] = np.sin(angle_rad)

        return True

    def save_volume(self):
        if self.volume is None:
            print("Error: No volume to save. Run reconstruction first.")
            return False

        print(f"Writing output to {self.output_file}")
        with mrcfile.new(self.output_file, overwrite=True) as mrc:
            # y 为变化最快的维度，所以先转置
            out_volume = np.transpose(self.volume, (1, 2, 0))
            mrc.set_data(out_volume)
            if hasattr(mrc, 'voxel_size'):
                mrc.voxel_size = self.pixel_size

        print(f"Reconstruction saved to {self.output_file}")
        return True

    def _weight_projections(self):
        """
        对 tilt 投影数据加权：乘以 tilt 角的余弦，并除以体积深度
        """
        if self.tilt_series is None:
            print("Error: No tilt series loaded. Call load_data() first.")
            return None

        weighted_tilt_series = np.zeros_like(self.tilt_series)
        for i, (cos_val, _) in enumerate(self.cos_sin_values):
            weighted_tilt_series[i] = self.tilt_series[i] * cos_val / self.vol_z
        return weighted_tilt_series

    def downsample_volume(self, downsample_factor=0.25):
        """
        对重建后的体积进行下采样，默认为 0.25（即每个维度的采样步长为 int(1/downsample_factor)）
        如果需要更精确的下采样，可以考虑其他插值方法
        """
        if self.volume is None:
            print("Error: No volume available for downsampling.")
            return

        if downsample_factor >= 1.0:
            print("Downsample factor >= 1.0, skipping downsampling.")
            return

        print(f"Downsampling volume by a factor of {downsample_factor}")
        step = int(1.0 / downsample_factor)
        self.volume = self.volume[::step, ::step, ::step]

    def reconstruct(self):
        raise NotImplementedError("Subclasses must implement this method")

class WBPReconstructor(TomoReconstructor):
    def __init__(self, input_file, output_file, tilt_file, vol_z=None, r_factor=4.0):
        super().__init__(input_file, output_file, tilt_file, vol_z)
        self.r_factor = r_factor

    def reconstruct(self):
        if self.tilt_series is None:
            if not self.load_data():
                return None

        self.volume = np.zeros((self.proj_x, self.vol_z, self.proj_y), dtype=np.float32)
        weighted_projections = self._weight_projections()

        start_time = time.time()
        print("Starting WBP reconstruction...")
        for y in tqdm(range(self.proj_y), desc="WBP Reconstruction"):
            sinogram = weighted_projections[:, y, :].astype(np.float32)
            filtered_sinogram = self._r_weight(sinogram)
            volume_slice = self._backproject(filtered_sinogram)
            self.volume[:, :, y] = volume_slice

        elapsed_time = time.time() - start_time
        print(f"WBP reconstruction completed in {elapsed_time:.2f} seconds")
        return self.volume

    def _r_weight(self, sinogram):
        """
        对 sinogram 进行 Ram-Lak 滤波（结合 Hamming 窗函数）
        """
        n_tilts, proj_x = sinogram.shape
        filtered_sinogram = np.zeros_like(sinogram)
        for i in range(n_tilts):
            projection = sinogram[i]
            fft_proj = fft.rfft(projection)
            ramp = np.fft.rfftfreq(proj_x)
            smoothing = 0.55 + 0.45 * np.cos(2 * np.pi * ramp)
            ramp = 2 * ramp * smoothing * self.r_factor
            filtered_fft = fft_proj * ramp
            filtered_proj = fft.irfft(filtered_fft, n=proj_x)
            filtered_sinogram[i] = filtered_proj
        return filtered_sinogram

    def _backproject(self, filtered_sinogram):
        volume_slice, count_slice = _backproject_numba(filtered_sinogram, 
                                                       self.cos_sin_values, 
                                                       self.proj_x, 
                                                       self.vol_z)
        mask = count_slice > 0
        volume_slice[mask] /= count_slice[mask]
        return volume_slice[:, ::-1]

class SIRTReconstructor(TomoReconstructor):
    def __init__(self, input_file, output_file, tilt_file, vol_z=None, iterations=10, 
                 relaxation=1.0, positivity=True, initial_volume=None):
        super().__init__(input_file, output_file, tilt_file, vol_z)
        self.iterations = iterations
        self.relaxation = relaxation
        self.positivity = positivity
        self.initial_volume = initial_volume

    def reconstruct(self):
        if self.tilt_series is None:
            if not self.load_data():
                return None

        if self.initial_volume is not None:
            print("Using provided initial volume")
            self.volume = self.initial_volume.copy()
        else:
            print("Starting with zero-initialized volume")
            self.volume = np.zeros((self.proj_x, self.vol_z, self.proj_y), dtype=np.float32)

        weighted_projections = self._weight_projections()
        start_time = time.time()
        print(f"Starting SIRT reconstruction with {self.iterations} iterations...")
        for iteration in range(self.iterations):
            update_volume = np.zeros_like(self.volume)
            weight_volume = np.zeros_like(self.volume)
            for y in tqdm(range(self.proj_y), desc=f"Iteration {iteration+1}/{self.iterations}"):
                sinogram = weighted_projections[:, y, :]
                synthetic_sinogram = self._forward_project(self.volume[:, :, y])
                diff_sinogram = sinogram - synthetic_sinogram
                slice_update, slice_weight = self._backproject_diff(diff_sinogram)
                update_volume[:, :, y] = slice_update
                weight_volume[:, :, y] = slice_weight

            mask = weight_volume > 0
            update_volume[mask] /= weight_volume[mask]
            self.volume[mask] += self.relaxation * update_volume[mask]
            if self.positivity:
                self.volume[self.volume < 0] = 0
            mean_update = np.mean(np.abs(update_volume[mask]))
            print(f"  Mean update: {mean_update:.6f}")
            if mean_update < 1e-6:
                print(f"Converged after {iteration+1} iterations.")
                break

        elapsed_time = time.time() - start_time
        print(f"SIRT reconstruction completed in {elapsed_time:.2f} seconds")
        return self.volume

    def _forward_project(self, volume_slice):
        return _forward_project_numba(volume_slice, self.cos_sin_values, self.proj_x)

    def _backproject_diff(self, diff_sinogram):
        update_slice, weight_slice = _backproject_numba(diff_sinogram, 
                                                       self.cos_sin_values, 
                                                       self.proj_x, 
                                                       self.vol_z)
        return update_slice[:, ::-1], weight_slice[:, ::-1]

class HybridReconstructor(TomoReconstructor):
    def __init__(self, input_file, output_file, tilt_file, vol_z=None, r_factor=4.0,
                 iterations=10, relaxation=1.0, positivity=True):
        super().__init__(input_file, output_file, tilt_file, vol_z)
        self.r_factor = r_factor
        self.iterations = iterations
        self.relaxation = relaxation
        self.positivity = positivity

    def reconstruct(self):
        if self.tilt_series is None:
            if not self.load_data():
                return None

        print("Phase 1: Performing WBP reconstruction...")
        wbp = WBPReconstructor(self.input_file, None, self.tilt_file, self.vol_z, self.r_factor)
        wbp.tilt_series = self.tilt_series
        wbp.tilt_angles = np.loadtxt(self.tilt_file)
        wbp.cos_sin_values = self.cos_sin_values
        wbp.pixel_size = self.pixel_size
        wbp.proj_x = self.proj_x
        wbp.proj_y = self.proj_y
        wbp_volume = wbp.reconstruct()

        print("Phase 2: Refining with SIRT...")
        sirt = SIRTReconstructor(self.input_file, None, self.tilt_file, self.vol_z,
                                 self.iterations, self.relaxation, 
                                 self.positivity, wbp_volume)
        sirt.tilt_series = self.tilt_series
        sirt.tilt_angles = np.loadtxt(self.tilt_file)
        sirt.cos_sin_values = self.cos_sin_values
        sirt.pixel_size = self.pixel_size
        sirt.proj_x = self.proj_x
        sirt.proj_y = self.proj_y
        self.volume = sirt.reconstruct()
        return self.volume

@numba.jit(nopython=True, parallel=True)
def _forward_project_numba(volume_slice, cos_sin_values, proj_x, scale=1):
    n_tilts = cos_sin_values.shape[0]
    vol_z = volume_slice.shape[1]
    proj_center_x = proj_x / 2.0
    vol_center_x = proj_x / 2.0
    vol_center_z_phys = (vol_z * scale) / 2.0
    sinogram = np.zeros((n_tilts, proj_x), dtype=np.float32)
    for i in range(n_tilts):
        cos_val = cos_sin_values[i, 0]
        sin_val = cos_sin_values[i, 1]
        ray_length = int(proj_x * abs(sin_val) + (vol_z * scale) * cos_val)
        for x_proj in range(proj_x):
            x_proj_centered = x_proj - proj_center_x
            ray_sum = 0.0
            ray_count = 0
            for step in range(ray_length):
                t = step - ray_length / 2.0
                x_vol = x_proj_centered * cos_val - t * sin_val + vol_center_x
                z_phys = x_proj_centered * sin_val + t * cos_val + vol_center_z_phys
                z_vol = z_phys / scale
                if 0 <= x_vol < proj_x and 0 <= z_vol < vol_z:
                    x_floor = int(x_vol)
                    z_floor = int(z_vol)
                    if x_floor < proj_x - 1 and z_floor < vol_z - 1:
                        x_alpha = x_vol - x_floor
                        z_alpha = z_vol - z_floor
                        val00 = volume_slice[x_floor, z_floor]
                        val01 = volume_slice[x_floor, z_floor + 1]
                        val10 = volume_slice[x_floor + 1, z_floor]
                        val11 = volume_slice[x_floor + 1, z_floor + 1]
                        
                        if val00 <= -1e10 or val01 <= -1e10 or val10 <= -1e10 or val11 <= -1e10:
                            continue
                        value = (1 - x_alpha) * (1 - z_alpha) * val00 + \
                                (1 - x_alpha) * z_alpha * val01 + \
                                x_alpha * (1 - z_alpha) * val10 + \
                                x_alpha * z_alpha * val11
                        ray_sum += value
                        ray_count += 1
            if ray_count > 0:
                sinogram[i, x_proj] = ray_sum / ray_count
            else:
                sinogram[i, x_proj] = 0.0
    return sinogram

@numba.jit(nopython=True, parallel=True)
def _backproject_numba(sinogram, cos_sin_values, proj_x, vol_z, scale=1):
    n_tilts = sinogram.shape[0]
    volume_slice = np.zeros((proj_x, vol_z), dtype=np.float32)
    count_slice = np.zeros((proj_x, vol_z), dtype=np.int32)
    proj_center_x = proj_x / 2.0
    vol_center_x = proj_x / 2.0
    vol_center_z_phys = (vol_z * scale) / 2.0
    for x in numba.prange(proj_x):
        x_centered = x - vol_center_x
        for z in range(vol_z):
            z_phys = z * scale
            z_centered_phys = z_phys - vol_center_z_phys
            for i in range(n_tilts):
                cos_val = cos_sin_values[i, 0]
                sin_val = cos_sin_values[i, 1]
                x_proj = x_centered * cos_val + z_centered_phys * sin_val + proj_center_x
                if 0 <= x_proj < proj_x - 1:
                    x_floor = int(x_proj)
                    x_alpha = x_proj - x_floor
                    val1 = sinogram[i, x_floor]
                    val2 = sinogram[i, x_floor + 1]
                    if val1 <= -1e10 or val2 <= -1e10:
                        continue
                    value = (1 - x_alpha) * val1 + x_alpha * val2
                    volume_slice[x, z] += value
                    count_slice[x, z] += 1
    return volume_slice, count_slice


def main():
    parser = argparse.ArgumentParser(description='Tomographic Reconstruction')
    parser.add_argument('-i', '--input', required=True, help='Input MRC file with tilt series')
    parser.add_argument('-o', '--output', required=True, help='Output MRC file for reconstructed volume')
    parser.add_argument('--method', choices=['wbp', 'sirt', 'hybrid'], default='hybrid',
                        help='Reconstruction method: wbp, sirt, or hybrid (default: hybrid)')
    parser.add_argument('-t', '--tilt_file', required=True, help='Tilt angle file, each line is an angle in degree')
    parser.add_argument('-z', '--vol_z', type=int, help='Z dimension of output volume (default: same as Y dimension)')
    parser.add_argument('--r-factor', type=float, default=4.0, 
                        help='Weighting factor for the Ram-Lak filter in WBP (default: 4.0)')
    parser.add_argument('--iterations', type=int, default=10, 
                        help='Number of iterations for SIRT (default: 10)')
    parser.add_argument('--relaxation', type=float, default=1.0, 
                        help='Relaxation parameter for SIRT (0-1, default: 1.0)')
    parser.add_argument('--no-positivity', action='store_false', dest='positivity',
                        help='Disable positivity constraint for SIRT (default: enabled)')
    parser.add_argument('--downsample_factor', type=float, default=0.25,
                        help='Downsample factor for the reconstructed volume (default: 0.25)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.method == 'wbp':
        print("Using Weighted Back-Projection (WBP) reconstruction")
        reconstructor = WBPReconstructor(
            args.input, 
            args.output,
            args.tilt_file,
            args.vol_z,
            args.r_factor
        )
    elif args.method == 'sirt':
        print("Using Simultaneous Iterative Reconstruction Technique (SIRT)")
        reconstructor = SIRTReconstructor(
            args.input, 
            args.output,
            args.tilt_file,
            args.vol_z,
            args.iterations,
            args.relaxation,
            args.positivity
        )
    else:  # hybrid
        print("Using Hybrid reconstruction (WBP initialization + SIRT refinement)")
        reconstructor = HybridReconstructor(
            args.input, 
            args.output,
            args.tilt_file,
            args.vol_z,
            args.r_factor,
            args.iterations,
            args.relaxation,
            args.positivity
        )

    reconstructor.load_data()
    reconstructor.reconstruct()
    
    # 调用封装好的下采样函数
    reconstructor.downsample_volume(args.downsample_factor)
    
    reconstructor.save_volume()
    print("Reconstruction completed successfully!")

if __name__ == '__main__':
    main()
