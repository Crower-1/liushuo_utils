#!/usr/bin/env python3
"""Generate two-color ChimeraX vesicle rendering by fitted axis ratio.

Example:
    python chimerax/generate_vesicle_shape_render.py \
        --json-path vesicles.json --mrc-path labels.mrc.7z \
        --output-dir render_vesicle_shape

Requires the neighboring generate_vesicle_render.py and its dependencies.
JSON radii are fitted semiaxes in isotropic voxels. Their max/min ratio
equals the full long/short axis ratio. Ellipsoidal means strictly > 1.14.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from generate_vesicle_render import (
    extract_mrc, load_vesicles, render_scripts, verify_cmms, write_cmms,
)


def classify(vesicles, entries, threshold):
    groups = [
        {'group': 1, 'file': '1.cmm', 'shape': 'spherical',
         'rgb': [0.0, 0.54, 0.54], 'vesicle_ids': [], 'markers': 0},
        {'group': 2, 'file': '2.cmm', 'shape': 'ellipsoidal',
         'rgb': [0.86, 0.08, 0.24], 'vesicle_ids': [], 'markers': 0},
    ]
    by_id, rows = {}, []
    for entry in entries:
        vid = int(entry['name'].split('_')[-1])
        radii = np.asarray(entry['radii'], dtype=float)
        ratio = float(radii.max() / radii.min())
        k = int(ratio > threshold)
        by_id[vid] = k
        groups[k]['vesicle_ids'].append(vid)
        rows.append({'vesicle_id': vid, 'short_radius_voxels': float(radii.min()),
                     'long_radius_voxels': float(radii.max()),
                     'axis_ratio': ratio, 'shape': groups[k]['shape']})
    if set(by_id) != set(vesicles):
        raise ValueError('Classification and validated JSON IDs differ')
    return groups, by_id, sorted(rows, key=lambda row: row['vesicle_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json-path', required=True, type=Path)
    parser.add_argument('--mrc-path', required=True, type=Path, help='Instance labels, not a binary membrane mask')
    parser.add_argument('--output-dir', required=True, type=Path, help='New or empty output directory')
    parser.add_argument('--ratio-threshold', type=float, default=1.14)
    parser.add_argument('--pixel-size', type=float, default=17.14, help='Isotropic voxel size in Angstroms')
    parser.add_argument('--model-id', type=int, default=15)
    parser.add_argument('--distance', type=float, default=40)
    parser.add_argument('--level', type=float, default=112)
    parser.add_argument('--marker-radius', type=float, default=10)
    args = parser.parse_args()
    try:
        values = [args.ratio_threshold, args.pixel_size, args.distance, args.level, args.marker_radius]
        if not np.all(np.isfinite(values)) or min(args.pixel_size, args.distance, args.marker_radius) <= 0:
            raise ValueError('Numeric arguments must be finite; pixel size, distance and marker radius must be positive')
        if args.ratio_threshold < 1 or args.model_id < 4:
            raise ValueError('Ratio threshold must be >=1 and model ID must be >=4')
        for name in ('json_path', 'mrc_path', 'output_dir'):
            path = getattr(args, name).expanduser().resolve()
            if any(c in str(path) for c in ('"', '\n', '\r', '\\')):
                raise ValueError('Paths with quotes, backslashes or newlines are unsupported')
            setattr(args, name, path)
        if not args.json_path.is_file() or not args.mrc_path.is_file():
            raise ValueError('Both inputs must exist')
        output = args.output_dir
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Output directory must be new or empty')
        vesicles = load_vesicles(args.json_path)
        doc = json.loads(args.json_path.read_text())
        groups, by_id, rows = classify(vesicles, doc.get('instances', doc.get('vesicles')), args.ratio_threshold)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vesicle-shape-mrc-') as scratch, tempfile.TemporaryDirectory(prefix='.vesicle-shape-', dir=output.parent) as staging:
            stage = Path(staging)
            print('Reading instance labels and generating CMMs...', flush=True)
            mrc = extract_mrc(args.mrc_path, Path(scratch))
            count, origin, mismatches = write_cmms(mrc, vesicles, groups, by_id, stage, args)
            verify_cmms(stage, groups)
            active = render_scripts(stage, output, groups, args)
            # Reuse the checked rendering workflow with shape-specific model names.
            for filename in ('render_vesicles.py', 'render_vesicles.cxc'):
                path = stage / filename
                text = path.read_text().replace('radius_group_1', 'spherical').replace('radius_group_2', 'ellipsoidal')
                text = text.replace('radius_group_{group}', "{('spherical' if group == 1 else 'ellipsoidal')}")
                text = text.replace('radius group', 'shape group').replace('Radius group', 'Shape group')
                if filename.endswith('.py'):
                    compile(text, filename, 'exec')
                path.write_text(text)
            for g in groups:
                path = stage / g['file']
                # Only change the opening marker-set line; CMMs can be large.
                with path.open('r+') as f:
                    # Preserve length so the remaining marker stream stays intact.
                    old = f.readline()
                    new = old.replace(f'radius_group_{g["group"]:02d}', g['shape'])
                    f.seek(0)
                    f.write(new.rstrip('\n').ljust(len(old) - 1) + '\n')
            with (stage / 'vesicle_shapes.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = {'source_json': str(args.json_path), 'source_mrc': str(args.mrc_path),
                      'criterion': 'max(radii) / min(radii) > threshold => ellipsoidal; otherwise spherical',
                      'ratio_threshold': args.ratio_threshold, 'total_vesicles': len(vesicles),
                      'total_markers': count, 'pixel_size_angstrom': args.pixel_size,
                      'origin_angstrom': origin, 'active_groups': active, 'groups': groups,
                      'result_parent_model': args.model_id, 'distance_angstrom': args.distance,
                      'render_level': args.level, 'center_id_mismatches': mismatches}
            (stage / 'groups.json').write_text(json.dumps(report, indent=2) + '\n')
            (stage / 'README.txt').write_text(
                f'在 ChimeraX 中先将对应灰度渲染图（如 *_reversed.mrc）打开为 #1，再执行：\n'
                f'open "{output / "render_vesicles.py"}"\n\n'
                f'长轴/短轴 > {args.ratio_threshold:g}：椭球，红色，2.cmm；否则：近球形，青色，1.cmm。\n'
                f'近球形 {len(groups[0]["vesicle_ids"])} 个；椭球 {len(groups[1]["vesicle_ids"])} 个。\n'
                f'#2、#3、#{args.model_id} 必须空闲。结果为 #{args.model_id}.1 和 #{args.model_id}.2；空组跳过。\n'
                f'distance={args.distance:g} Å，level={args.level:g}。逐囊泡分类见 vesicle_shapes.csv。\n'
                '输入标签按 ID 对应 JSON，不使用二值膜图。原始数据保持不变。\n'
                '生成器不执行 ChimeraX 渲染，不自动保存会话。\n')
            output.mkdir(exist_ok=True)
            if any(output.iterdir()):
                raise ValueError('Output became nonempty during generation')
            for path in stage.iterdir():
                shutil.move(str(path), output / path.name)
        print(f'Saved to {output}; spherical={len(groups[0]["vesicle_ids"])}, ellipsoidal={len(groups[1]["vesicle_ids"])}, markers={count}')
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
