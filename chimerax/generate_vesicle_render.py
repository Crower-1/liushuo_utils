#!/usr/bin/env python3
"""Generate radius-grouped CMMs and ChimeraX scripts from vesicle JSON + MRC.

Example:
    python generate_vesicle_render.py --json-path vesicles.json \
        --mrc-path labels.mrc.7z --output-dir render_vesicle --model-id 14

Requires numpy, mrcfile, and (for .7z) libarchive-c or the bsdtar executable.
JSON centers/radii are in voxels; pixel size is in Angstroms; bin radii in nm.
The input MRC is an instance label map, not the grayscale rendering image.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import mrcfile
import numpy as np


def extract_mrc(source, directory):
    """Extract just one MRC to a controlled path, never archive-owned paths."""
    if source.suffix.lower() == '.mrc':
        return source
    if source.suffix.lower() != '.7z':
        raise ValueError('MRC input must be .mrc or .mrc.7z')
    target = directory / 'labels.mrc'
    try:
        import libarchive
    except ImportError:
        libarchive = None
    if libarchive is not None:
        found = []
        with libarchive.file_reader(str(source)) as archive:
            for entry in archive:
                if entry.pathname.lower().endswith('.mrc'):
                    if not entry.isfile:
                        raise ValueError('MRC archive member must be a regular file')
                    found.append(entry.pathname)
                    if len(found) > 1:
                        raise ValueError('Archive must contain exactly one MRC file')
                    with target.open('wb') as out:
                        for block in entry.get_blocks():
                            out.write(block)
        if not found:
            raise ValueError('No MRC found in archive')
    else:
        tar = shutil.which('bsdtar')
        if not tar:
            raise ValueError('Install libarchive-c or bsdtar to read .7z archives')
        names = subprocess.check_output([tar, '-tf', str(source)], text=True).splitlines()
        members = [n for n in names if n.lower().endswith('.mrc')]
        if len(members) != 1:
            raise ValueError('Archive must contain exactly one MRC file')
        with target.open('wb') as out:
            subprocess.run([tar, '-xOf', str(source), '--', members[0]], stdout=out, check=True)
    return target


def load_vesicles(path):
    doc = json.loads(path.read_text())
    entries = doc.get('instances', doc.get('vesicles'))
    if not isinstance(entries, list) or not entries:
        raise ValueError('JSON must contain a nonempty instances or vesicles list')
    result = {}
    for entry in entries:
        match = re.fullmatch(r'vesicle_(\d+)', entry.get('name', ''))
        if not match:
            raise ValueError(f"Invalid vesicle name: {entry.get('name')!r}")
        vid = int(match.group(1))
        radii = np.asarray(entry['radii'], dtype=float)
        center = np.asarray(entry['center'], dtype=float)
        if vid <= 0 or vid in result:
            raise ValueError(f'Duplicate or nonpositive vesicle ID: {vid}')
        if radii.shape != (3,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError(f'Invalid radii for vesicle {vid}')
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError(f'Invalid center for vesicle {vid}')
        result[vid] = {'radius_voxels': float(radii.mean()), 'center': center}
    return result


def make_groups(vesicles, args):
    radii = {i: v['radius_voxels'] * args.pixel_size / 10 for i, v in vesicles.items()}
    lo = args.min_radius_nm
    hi = args.max_radius_nm if args.max_radius_nm is not None else max(lo, max(radii.values()))
    if hi < lo:
        raise ValueError('Maximum radius must be >= minimum radius')
    edges = np.linspace(lo, hi, args.bins + 1)
    red, middle, teal = np.array([[.86, .08, .24], [.49, .99, 0], [0, .54, .54]])
    groups = []
    for k, t in enumerate(np.linspace(0, 1, args.bins)):
        rgb = red * (1 - 2*t) + middle * (2*t) if t <= .5 else middle * (2 - 2*t) + teal * (2*t - 1)
        groups.append({'group': k + 1, 'file': f'{k + 1}.cmm',
                       'radius_nm': [float(edges[k]), float(edges[k + 1])],
                       'rgb': rgb.tolist(), 'vesicle_ids': [], 'markers': 0})
    by_id = {}
    for vid, r in sorted(radii.items()):
        k = int(np.clip(np.searchsorted(edges, r, side='right') - 1, 0, args.bins - 1)) if hi > lo else 0
        groups[k]['vesicle_ids'].append(vid)
        by_id[vid] = k
    return groups, by_id, [lo, hi]


def write_cmms(mrc_path, vesicles, groups, by_id, stage, args):
    with mrcfile.mmap(str(mrc_path), mode='r') as mrc, ExitStack() as stack:
        data = mrc.data
        if data.ndim != 3 or not np.issubdtype(data.dtype, np.integer):
            raise ValueError('MRC must be a 3D integer instance label map')
        if tuple(int(mrc.header[k]) for k in ('mapc', 'mapr', 'maps')) != (1, 2, 3):
            raise ValueError('Only standard MRC axes (mapc,mapr,maps = 1,2,3) are supported')
        if not np.allclose(mrc.voxel_size.tolist(), [args.pixel_size]*3, rtol=1e-4):
            raise ValueError(f'MRC voxel size {mrc.voxel_size} differs from --pixel-size {args.pixel_size}')
        if any(int(mrc.header[k]) for k in ('nxstart', 'nystart', 'nzstart')):
            raise ValueError('Nonzero MRC start indices are unsupported; normalize map coordinates first')
        origin = tuple(float(x) for x in mrc.header.origin.tolist())
        if not np.all(np.isfinite(origin)):
            raise ValueError('Invalid MRC origin')
        ids = set()
        foreground = 0
        for plane in data:
            ids.update(map(int, np.unique(plane)))
            foreground += int(np.count_nonzero(plane))
        ids.discard(0)
        if ids != set(vesicles):
            raise ValueError(f'ID mismatch: absent from JSON={sorted(ids-set(vesicles))}; '
                             f'absent from MRC={sorted(set(vesicles)-ids)}')
        # IDs define masks; fitted centers may lie outside irregular/hollow masks.
        center_mismatches = []
        for vid, vesicle in vesicles.items():
            center = np.rint(vesicle['center']).astype(int)
            if np.any(center < 0) or np.any(center >= data.shape) or int(data[tuple(center)]) != vid:
                center_mismatches.append(vid)
        files = []
        for g in groups:
            f = stack.enter_context((stage / g['file']).open('w'))
            f.write(f'<marker_set name="radius_group_{g["group"]:02d}">\n')
            files.append(f)
        color_strings = [' '.join(f'{c}="{v:.6f}"' for c, v in zip(('r', 'g', 'b'), g['rgb'])) for g in groups]
        for z, plane in enumerate(data):
            yy, xx = np.where(plane != 0)
            for y, x in zip(yy, xx):
                k = by_id[int(plane[y, x])]
                g = groups[k]
                g['markers'] += 1
                files[k].write(f'<marker id="{g["markers"]}" '
                               f'x="{x*args.pixel_size+origin[0]:.2f}" '
                               f'y="{y*args.pixel_size+origin[1]:.2f}" '
                               f'z="{z*args.pixel_size+origin[2]:.2f}" '
                               f'{color_strings[k]} radius="{args.marker_radius:g}"/>\n')
        for f in files:
            f.write('</marker_set>\n')
        if sum(g['markers'] for g in groups) != foreground:
            raise ValueError('Foreground marker count mismatch')
    return foreground, origin, center_mismatches


def verify_cmms(stage, groups):
    for g in groups:
        count = 0
        for _, element in ET.iterparse(stage / g['file'], events=('end',)):
            if element.tag == 'marker':
                count += 1
                if int(element.attrib['id']) != count:
                    raise ValueError('CMM marker IDs are not sequential')
                expected = tuple(f'{v:.6f}' for v in g['rgb'])
                if tuple(element.attrib[c] for c in ('r', 'g', 'b')) != expected:
                    raise ValueError('CMM color mismatch')
            element.clear()
        if count != g['markers']:
            raise ValueError('CMM marker count mismatch')


def render_scripts(stage, output, groups, args):
    active = [g['group'] for g in groups if g['markers']]
    config = {'groups': active, 'model_id': args.model_id, 'distance': args.distance, 'level': args.level}
    runtime = '''"""Run this script inside ChimeraX, not system Python."""
from pathlib import Path
from chimerax.core.commands import run
from chimerax.core.errors import UserError
from chimerax.map import Volume

CONFIG = CONFIG_PLACEHOLDER
folder = Path(__file__).resolve().parent
models = session.models.list()
source = next((m for m in models if m.id == (1,)), None)
if not isinstance(source, Volume):
    raise UserError("Open the grayscale rendering map (for example *_reversed.mrc) as #1 first.")
occupied = [m.id_string for m in models if m.id and m.id[0] in (2, 3, CONFIG['model_id'])]
if occupied:
    raise UserError("Temporary/output model IDs are occupied: " + ", ".join(occupied))
for group in CONFIG['groups']:
    if not (folder / f"{group}.cmm").is_file():
        raise UserError(f"Missing CMM for group {group}")
run(session, "volume #1 style surface")
for index, group in enumerate(CONFIG['groups'], 1):
    session.logger.info(f"Rendering radius group {group} ({index}/{len(CONFIG['groups'])})")
    try:
        run(session, f'open "{folder / f"{group}.cmm"}" id #2')
        run(session, f"color zone #1 near #2 distance {CONFIG['distance']}")
        run(session, "vop splitbyzone #1")
        children = {m.id for m in session.models.list() if m.id and m.id[0] == 3}
        if (3, 1) not in children or (3, 2) not in children or any(len(i)>1 and i[1]>2 for i in children):
            raise UserError("Expected #3.1 background and #3.2 single-color zone; stopping for inspection.")
        run(session, "close #2#3.1")
        run(session, f"volume #3.2 level {CONFIG['level']}")
        run(session, f"rename #3 radius_group_{group} id #{CONFIG['model_id']}.{group}")
    except Exception as error:
        raise UserError(f"Stopped at radius group {group}; completed results are retained. {error}") from error
run(session, "hide #1 models")
run(session, f"show #{CONFIG['model_id']} models")
session.logger.info("Radius group rendering complete; save the session if needed.")
'''.replace('CONFIG_PLACEHOLDER', repr(config))
    compile(runtime, 'render_vesicles.py', 'exec')
    (stage / 'render_vesicles.py').write_text(runtime)
    commands = ['# #1 = grayscale map. #2, #3 and output parent must be free.',
                '# Source sampling and region are preserved. Empty bins are skipped.', 'volume #1 style surface']
    for k in active:
        commands.extend([f'open "{output / f"{k}.cmm"}" id #2',
                         f'color zone #1 near #2 distance {args.distance:g}', 'vop splitbyzone #1',
                         'close #2#3.1', f'volume #3.2 level {args.level:g}',
                         f'rename #3 radius_group_{k} id #{args.model_id}.{k}', ''])
    commands.extend(['hide #1 models', f'show #{args.model_id} models'])
    (stage / 'render_vesicles.cxc').write_text('\n'.join(commands) + '\n')
    return active


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json-path', required=True, type=Path)
    parser.add_argument('--mrc-path', required=True, type=Path, help='Instance label .mrc.7z or .mrc')
    parser.add_argument('--output-dir', required=True, type=Path, help='New or empty render_vesicle folder')
    parser.add_argument('--pixel-size', type=float, default=17.14, help='Isotropic voxel size in Angstroms (17.14)')
    parser.add_argument('--bins', type=int, default=8)
    parser.add_argument('--min-radius-nm', type=float, default=15)
    parser.add_argument('--max-radius-nm', type=float, help='Default: maximum mean radius in this dataset')
    parser.add_argument('--model-id', type=int, default=14, help='ChimeraX output parent ID (14); must be >=4')
    parser.add_argument('--distance', type=float, default=40, help='ChimeraX color-zone distance in Angstroms')
    parser.add_argument('--level', type=float, default=112, help='Isosurface threshold of grayscale rendering map')
    parser.add_argument('--marker-radius', type=float, default=10, help='CMM marker radius in Angstroms')
    args = parser.parse_args()
    try:
        if args.bins < 2 or args.model_id < 4:
            raise ValueError('--bins must be >=2 and --model-id must be >=4')
        numbers = [args.pixel_size, args.min_radius_nm, args.distance, args.level, args.marker_radius]
        if args.max_radius_nm is not None:
            numbers.append(args.max_radius_nm)
        if not np.all(np.isfinite(numbers)) or min(args.pixel_size, args.distance, args.marker_radius) <= 0 or args.min_radius_nm < 0:
            raise ValueError('Invalid numeric arguments')
        for name in ('json_path', 'mrc_path', 'output_dir'):
            path = getattr(args, name).expanduser().resolve()
            if any(c in str(path) for c in ('"', '\n', '\r', '\\')):
                raise ValueError('Paths with quotes, backslashes or newlines are unsupported by generated commands')
            setattr(args, name, path)
        if not args.json_path.is_file() or not args.mrc_path.is_file():
            raise ValueError('Both input files must exist')
        output = args.output_dir
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Output directory must be new or empty; existing results are never overwritten')
        vesicles = load_vesicles(args.json_path)
        groups, by_id, bounds = make_groups(vesicles, args)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vesicle-mrc-') as scratch, tempfile.TemporaryDirectory(prefix='.vesicle-render-', dir=output.parent) as staging:
            stage = Path(staging)
            print('Reading label MRC...', flush=True)
            mrc_path = extract_mrc(args.mrc_path, Path(scratch))
            count, origin, center_mismatches = write_cmms(mrc_path, vesicles, groups, by_id, stage, args)
            print('Validating CMM files...', flush=True)
            verify_cmms(stage, groups)
            active = render_scripts(stage, output, groups, args)
            report = {'source_json': str(args.json_path), 'source_mrc': str(args.mrc_path),
                      'pixel_size_angstrom': args.pixel_size, 'origin_angstrom': origin,
                      'mean_radius_nm_bounds': bounds, 'total_vesicles': len(vesicles), 'total_markers': count,
                      'binning': 'Equal-width mean-radius bins, left-inclusive; final right edge included; outliers clamped. If both bounds coincide, all go to bin 1.',
                      'palette': 'red -> yellowgreen -> teal', 'active_groups': active,
                      'result_parent_model': args.model_id, 'distance_angstrom': args.distance,
                      'render_level': args.level, 'center_id_mismatches': center_mismatches, 'groups': groups}
            (stage / 'groups.json').write_text(json.dumps(report, indent=2) + '\n')
            (stage / 'README.txt').write_text(
                f'在 ChimeraX 执行：\nopen "{output / "render_vesicles.py"}"\n\n'
                f'#1 为对应灰度渲染图（如 *_reversed.mrc），不是本次输入的囊泡标签图。\n'
                f'#2、#3、#{args.model_id} 必须空闲。空分组自动跳过。保留 #1 当前采样和范围。\n'
                f'distance {args.distance:g}，level {args.level:g}；结果为 #{args.model_id}.分组号，实际 map 为 #{args.model_id}.分组号.2。\n'
                f'{len(vesicles)} 个囊泡；各档数量：{[len(g["vesicle_ids"]) for g in groups]}。\n'
                '分组和坐标单位见 groups.json。生成器不执行 ChimeraX 渲染，不自动保存会话。\n')
            output.mkdir(exist_ok=True)
            if any(output.iterdir()):
                raise ValueError('Output directory became nonempty during generation')
            for path in stage.iterdir():
                shutil.move(str(path), output / path.name)
        print(f'Saved {len(groups)} CMMs, {len(vesicles)} vesicles, {count} markers to {output}')
        print(f'Group counts: {[len(g["vesicle_ids"]) for g in groups]}; {len(active)} rendering passes')
        if center_mismatches:
            print(f'Note: {len(center_mismatches)} fitted centers do not match label IDs; masks were matched by ID. See groups.json.')
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
