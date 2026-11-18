"""Torch-free paper figures and strictly selected image downloads.

All metrics are read from saved records. Rendering never changes measurements.
Example rows contain one target; the raw global peak decides which camera panels
are downloaded. No checkpoint, feature cache, or dataset snapshot is inspected.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

LAYERS = (1, 7, 13, 19, 25)
PARTITIONS = ('with_coordinates', 'without_coordinates')
CAMERAS = ('Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right')
IMAGE_REPO = 'minhdang0901/drivelm-nuscenes-v1-0'
IMAGE_REVISION = '6c4336a5279eaba3f3dfc5b525cc6ba57d55d61a'
_STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
          'font.size': 12, 'axes.labelsize': 14, 'axes.linewidth': 1.5,
          'axes.spines.top': False, 'axes.spines.right': False,
          'legend.frameon': False, 'pdf.fonttype': 42, 'text.usetex': False,
          'figure.facecolor': 'white', 'axes.facecolor': 'white'}
_CAUSAL_CAPTION = '''Causal intervention on post-encoder/connector visual embeddings,
not deletion of raw image regions. For each saved original response, the same
content-token IDs and denominator are teacher-forced in every condition.
Comp = -expm1(s_remove - s_base); Sgap = -expm1(s_keep - s_base), where s is
mean token log probability (geometric-mean probability scoring). Attention
ranking is separate at each layer; this differs from mean-probability scoring
and all-layer ranking in the reference causal-perturbation paper. Top/Bottom
select 32 of 384 positions; Random averages three independent repeats per
sample, shared across layers. Each point is one sample; negative values and
outliers are retained on common linear axes. Zeroing contextualized embeddings
does not guarantee removal of all information about an object.
'''


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def _sidecars(folder, name, caption, metadata):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / f'{name}_caption.txt').write_text(caption, encoding='utf-8')
    if metadata is not None:
        _json(Path(folder) / f'{name}_metadata.json', metadata)


def _export(fig, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    paths = [Path(folder) / f'{name}.{ext}' for ext in ('png', 'pdf')]
    try:
        for path in paths:
            fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
    finally:
        plt.close(fig)
    return paths


def _limits(values):
    low, high = float(np.min(values)), float(np.max(values))
    margin = (high - low) * .06 or max(abs(low) * .06, .05)
    return low - margin, high + margin


def plot_causal(records, output_dir, metadata=None) -> list[Path]:
    """Export five scatter plots with identical linear limits, including outliers."""
    records = list(records)
    points = {layer: {kind: [] for kind in ('top', 'random', 'bottom')}
              for layer in LAYERS}
    for record in records:
        layers = {item['layer']: item for item in record['layers']}
        for layer in LAYERS:
            for kind in ('top', 'random', 'bottom'):
                condition = record['random'] if kind == 'random' else layers[layer][kind]
                pair = [float(condition['comp']), float(condition['sgap'])]
                if not np.all(np.isfinite(pair)):
                    raise ValueError('Causal metrics must be finite; numerical failures cannot be omitted')
                points[layer][kind].append(pair)
    flat = np.asarray([p for layer in points.values() for values in layer.values()
                       for p in values], dtype=float).reshape(-1, 2)
    xlim, ylim = (_limits(flat[:, 0]), _limits(flat[:, 1])) if len(flat) else ((-.05, 1.05), (-.05, 1.05))
    paths = []
    with plt.rc_context(_STYLE):
        for layer in LAYERS:
            fig, ax = plt.subplots(figsize=(7.2, 5.4), layout='constrained')
            for kind, color in zip(('top', 'random', 'bottom'),
                                   ('#1f77b4', '#ff7f0e', '#2ca02c')):
                values = np.asarray(points[layer][kind], dtype=float).reshape(-1, 2)
                ax.scatter(values[:, 0], values[:, 1], s=12, alpha=.35,
                           color=color, edgecolors='none', label=kind.title(), rasterized=True)
            ax.set(xlabel='Normalized Comprehensiveness',
                   ylabel='Normalized Sufficiency Gap', xlim=xlim, ylim=ylim,
                   title=f'Layer {layer} (zero-based); N = {len(records):,}')
            ax.grid(alpha=.18, linewidth=.6); ax.set_axisbelow(True)
            ax.legend(markerscale=1.8)
            paths.extend(_export(fig, output_dir, f'causal_layer_{layer}'))
    _sidecars(output_dir, 'causal', _CAUSAL_CAPTION, metadata)
    return paths


def _partition_caption(partition, data):
    eligible, total = data['eligible_targets'], data['total_targets']
    coverage = f'{eligible / total:.1%}' if total else 'N/A'
    counts = ', '.join(f"{g['layers'][0]}–{g['layers'][1]}: {g['correct']}/{g['total']}"
                       for g in data['groups'])
    return (f'{partition}: coverage {eligible}/{total} ({coverage}); correct/eligible by layer: '
            f'{counts}. Bboxes without any grid center: total {data.get("unreachable_total", 0)}, '
            f'eligible {data.get("unreachable_eligible", 0)}. '
            f'Eligibility audit: {json.dumps(data.get("reasons", {}), sort_keys=True)}.\n')


def plot_localization(summary, output_dir, metadata=None) -> list[Path]:
    """Plot target-weighted global-peak localization; empty denominator is N/A."""
    caption = ('Localization counts targets whose global maximum over all 384 positions '
               'has the correct camera and a grid-cell center inside the target bbox. '
               'Mean heads, phrase tokens, then five layers before global argmax. '
               'Each partition uses the same eligible denominator for all six groups. '
               'Exact textual eligibility does not establish semantic object recognition.\n')
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5.4), layout='constrained')
        for index, partition in enumerate(PARTITIONS):
            data = summary[partition]
            groups = data['groups']
            if len(groups) != 6 or any(g['total'] != data['eligible_targets'] for g in groups):
                raise ValueError('All six layer groups must share the eligible-target denominator')
            values = [100 * g['correct'] / g['total'] if g['total'] else 0 for g in groups]
            bars = ax.bar(np.arange(6) + (index - .5) * .36, values, width=.35,
                          label=partition.replace('_', ' ').title(),
                          color=('#3775BA', '#E9A6A1')[index], edgecolor='#272727', linewidth=.8)
            labels = [f'{v:.1f}%' if g['total'] else 'N/A' for v, g in zip(values, groups)]
            ax.bar_label(bars, labels=labels, padding=4, fontsize=10)
            caption += _partition_caption(partition, data)
        ax.set_xticks(np.arange(6), [f'{5*g}–{5*g+4}' for g in range(6)])
        ax.set(xlabel='Layer group (zero-based)', ylabel='Localization accuracy (%)', ylim=(0, 112))
        ax.set_yticks([0, 25, 50, 75, 100]); ax.legend(loc='upper left', ncol=2, fontsize=10)
        paths = _export(fig, output_dir, 'localization_accuracy')
    _sidecars(output_dir, 'localization', caption, metadata)
    return paths


def _stable_key(seed, sample_id, target):
    value = json.dumps([seed, sample_id, target], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _eligible(target):
    return target.get('status') == 'eligible' or target.get('eligible') is True


def select_examples(records, summary, seed=42):
    """Choose at most three deterministic rows per partition, without image I/O.

    Returned rows have sample/frame IDs, paths, target_camera, group_index,
    layer_start/end, selection_reason, and a one-element targets list preserving
    the evaluated target and raw [6,384] maps. The best group is selected post hoc;
    earliest group wins ties. Scarce miss/small roles are filled before hit.
    """
    records = list(records)
    selected = {partition: [] for partition in PARTITIONS}
    for partition in PARTITIONS:
        data = summary[partition]
        if not data['eligible_targets']:
            continue
        group_index = max(range(6), key=lambda i: (data['groups'][i]['accuracy'] or 0, -i))
        candidates = []
        for record in records:
            if record.get('partition') != partition:
                continue
            for evaluated in record.get('targets', []):
                if not _eligible(evaluated):
                    continue
                target = evaluated['target']
                decision = evaluated['group_results'][group_index]
                row = {key: record.get(key, '') for key in
                       ('sample_id', 'frame_id', 'partition', 'question', 'response', 'camera_paths')}
                row.update(target_camera=target['camera'], group_index=group_index,
                           layer_start=decision['layer_start'], layer_end=decision['layer_end'],
                           targets=[evaluated])
                candidates.append((row, bool(decision['correct']),
                                   float(target.get('size_fraction', 1)),
                                   _stable_key(seed, record['sample_id'], target)))
        chosen = []; used = set()
        for role in ('miss', 'small', 'hit'):
            pool = [c for c in candidates if c[3] not in used and
                    (role == 'small' or c[1] == (role == 'hit'))]
            if not pool:
                continue
            frames = {c[0]['frame_id'] for c in chosen}
            cameras = {c[0]['target_camera'] for c in chosen}
            categories = {c[0]['targets'][0]['target'].get('category') for c in chosen}
            # Diversity is preferred when the requested role has alternatives.
            best = min(pool, key=lambda c: (
                c[0]['frame_id'] in frames,
                c[2] if role == 'small' else 0,
                c[0]['target_camera'] in cameras,
                c[0]['targets'][0]['target'].get('category') in categories, c[3]))
            best[0]['selection_reason'] = role
            chosen.append(best); used.add(best[3])
        while len(chosen) < min(3, len(candidates)):
            pool = [c for c in candidates if c[3] not in used]
            if not pool:
                break
            frames = {c[0]['frame_id'] for c in chosen}
            best = min(pool, key=lambda c: (c[0]['frame_id'] in frames, c[3]))
            best[0]['selection_reason'] = 'additional'; chosen.append(best); used.add(best[3])
        selected[partition] = [c[0] for c in sorted(chosen, key=lambda c:
            {'hit': 0, 'miss': 1, 'small': 2, 'additional': 3}[c[0]['selection_reason']])]
    return selected


def _peak_camera(decision):
    camera = decision.get('peak_camera')
    return CAMERAS[int(camera)] if isinstance(camera, (int, np.integer)) else camera


def _selected_paths(examples):
    paths = []
    for rows in examples.values():
        for row in rows:
            cameras = [row['target_camera']]
            cameras.extend(_peak_camera(t['group_results'][row['group_index']]) for t in row['targets'])
            for camera in dict.fromkeys(cameras):
                path = row['camera_paths'].get(camera)
                if path is not None and path not in paths:
                    paths.append(path)
    return paths


def _repo_path(path):
    if not isinstance(path, str) or '\\' in path or ':' in path or '?' in path or '#' in path:
        raise ValueError('Invalid dataset image path')
    path = path[5:] if path.startswith('data/nuscenes/') else path
    parts = PurePosixPath(path).parts
    if len(parts) < 3 or parts[0] != 'nuscenes' or '..' in parts or path.startswith('/'):
        raise ValueError('Image path must be inside nuscenes/')
    if PurePosixPath(path).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        raise ValueError('Unsupported image extension')
    return str(PurePosixPath(path))


def _valid_image(path):
    try:
        with Image.open(path) as source:
            source.verify()
        return True
    except (OSError, ValueError):
        return False


def _download(url, destination):
    request = urllib.request.Request(url, headers={'User-Agent': 'smolvlm-probing-figures/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response, open(destination, 'wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _retry_delay(error, attempt):
    header = getattr(error, 'headers', None)
    retry = header.get('Retry-After') if header else None
    if retry:
        try:
            return max(0., float(retry))
        except ValueError:
            try:
                return max(0., (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                pass
    return float(2 ** (attempt - 1))


def fetch_example_images(examples, image_dir, *, downloader=None, sleep=None):
    """Fetch only selected panels serially into a revision-specific atomic cache.

    ``downloader(url, temporary_path)`` may be injected for offline tests. Existing
    valid local original paths and this cache are used before network access.
    Returns (original-path -> local Path, missing audit list). HTTP 404 is final;
    429, server failures and transport errors get at most three attempts.
    """
    folder = Path(image_dir); folder.mkdir(parents=True, exist_ok=True)
    downloader = downloader or _download; sleep = sleep or time.sleep
    mapping, missing, receipts = {}, [], []
    resolved, failures = {}, {}
    for original in _selected_paths(examples):
        attempts = 0; repo_path = None
        try:
            repo_path = _repo_path(original)
            cached = folder / IMAGE_REVISION / Path(repo_path)
            if repo_path in resolved:
                mapping[original] = resolved[repo_path]
                continue
            if repo_path in failures:
                missing.append({**failures[repo_path], 'path': original})
                continue
            local = Path(original)
            if local.is_file() and _valid_image(local):
                mapping[original] = resolved[repo_path] = local
                receipts.append({'path': original, 'repo_path': repo_path, 'source': 'local'})
                continue
            if cached.is_file() and _valid_image(cached):
                mapping[original] = resolved[repo_path] = cached
                receipts.append({'path': original, 'repo_path': repo_path, 'source': 'cache'})
                continue
            cached.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_name(cached.name + '.partial')
            url = f'https://huggingface.co/datasets/{IMAGE_REPO}/resolve/{IMAGE_REVISION}/{urllib.parse.quote(repo_path, safe="/")}'
            for attempts in range(1, 4):
                try:
                    downloader(url, temporary)
                    if not _valid_image(temporary):
                        raise OSError('Downloaded file is not a valid image')
                    temporary.replace(cached)
                    mapping[original] = resolved[repo_path] = cached
                    receipts.append({'path': original, 'repo_path': repo_path,
                                     'source': 'download', 'attempts': attempts, 'url': url})
                    break
                except (OSError, urllib.error.URLError) as error:
                    temporary.unlink(missing_ok=True)
                    code = getattr(error, 'code', None)
                    if attempts == 3 or (code is not None and code != 429 and code < 500):
                        raise
                    sleep(_retry_delay(error, attempts))
        except (ValueError, OSError, urllib.error.URLError) as error:
            failure = {'path': original, 'repo_path': repo_path,
                       'error': str(error), 'attempts': attempts}
            missing.append(failure)
            if repo_path is not None:
                failures[repo_path] = failure
    receipt = {'repo_id': IMAGE_REPO, 'revision': IMAGE_REVISION,
               'selected_paths': _selected_paths(examples), 'available': receipts, 'missing': missing}
    temporary_receipt = folder / 'image_fetch_receipt.json.partial'
    _json(temporary_receipt, receipt); temporary_receipt.replace(folder / 'image_fetch_receipt.json')
    return mapping, missing


def _image_panel(ax, row, camera, image_paths):
    original = row['camera_paths'].get(camera)
    path = image_paths.get(original)
    target = row['targets'][0]['target']
    width, height = target['image_size']
    if path is not None:
        try:
            with Image.open(path) as source:
                pixels = np.asarray(source.convert('RGB'))
            ax.imshow(pixels, extent=(0, width, height, 0))
        except (OSError, ValueError):
            path = None
    if path is None:
        ax.text(.5, .5, f'Image unavailable\n{camera}', ha='center', va='center', transform=ax.transAxes)
    ax.set(xlim=(0, width), ylim=(height, 0)); ax.set_axis_off()
    return width, height


def plot_examples(examples, image_paths, output_dir, metadata=None) -> list[Path]:
    """Render post-hoc examples using raw grids and one shared log norm per montage."""
    paths = []
    cmap = LinearSegmentedColormap.from_list('blue_cyan_yellow_red',
                                            ['#0000ff', '#00ffff', '#ffff00', '#ff0000'])
    cmap = cmap.with_extremes(bad=(0, 0, 0, 0))
    with plt.rc_context(_STYLE):
        for partition in PARTITIONS:
            rows = examples.get(partition, [])
            if not rows:
                continue
            values = []
            for row in rows:
                for evaluated in row['targets']:
                    array = np.asarray(evaluated['attention_maps'], dtype=float)
                    if array.shape != (6, 384) or np.any(array < 0) or not np.all(np.isfinite(array)):
                        raise ValueError('Example attention maps must be finite nonnegative [6,384]')
                    values.extend(array[row['group_index']][array[row['group_index']] > 0])
            vmin, vmax = (min(values), max(values)) if values else (1e-6, 1.)
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.001))
            wrong = any(_peak_camera(t['group_results'][r['group_index']]) != r['target_camera']
                        for r in rows for t in r['targets'])
            columns = 3 if wrong else 2
            fig, axes = plt.subplots(len(rows), columns, figsize=(columns * 4.2, len(rows) * 3.8),
                                     squeeze=False, layout='constrained')
            # Reserve space for exact phrases below images and two-line titles
            # above the next row; the default constrained gap can overlap them.
            fig.get_layout_engine().set(h_pad=.12, hspace=.18)
            for row_index, row in enumerate(rows):
                # A row is one sample/target camera; selection currently keeps one map.
                evaluated = row['targets'][0]; target = evaluated['target']
                decision = evaluated['group_results'][row['group_index']]
                peak_camera = _peak_camera(decision)
                title = f"Averaged layers {row['layer_start']}–{row['layer_end']}"
                _image_panel(axes[row_index, 0], row, row['target_camera'], image_paths)
                axes[row_index, 0].set_title(f"(a) Original: {row['target_camera']}\n{row['sample_id']}", fontsize=10)
                array = np.asarray(evaluated['attention_maps'])[row['group_index']]
                cameras = [row['target_camera']]
                if peak_camera != row['target_camera']:
                    cameras.append(peak_camera)
                for column, camera in enumerate(cameras, start=1):
                    ax = axes[row_index, column]
                    width, height = _image_panel(ax, row, camera, image_paths)
                    grid = array[CAMERAS.index(camera)*64:(CAMERAS.index(camera)+1)*64].reshape(8, 8)
                    ax.imshow(np.ma.masked_equal(grid, 0), extent=(0, width, height, 0),
                              cmap=cmap, norm=norm, interpolation='bilinear', alpha=.55)
                    if camera == row['target_camera']:
                        x0, y0, x1, y1 = target['bbox']
                        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                               edgecolor='white', linewidth=1.8))
                    if camera == peak_camera:
                        # Use saved raw global 8x8 argmax, never an interpolated maximum.
                        peak = int(decision['peak_index']) % 64
                        py, px = divmod(peak, 8)
                        ax.plot((px+.5)*width/8, (py+.5)*height/8, '+', color='white',
                                markersize=12, markeredgewidth=2)
                    prefix = '(b) Overlay' if column == 1 else '(b) Global peak camera'
                    ax.set_title(f'{prefix}: {camera}\n{title}', fontsize=10)
                    phrase = target.get('reference_span', {}).get('text', '')
                    ax.text(.5, -.07, f'“{phrase}”', ha='center', va='top',
                            transform=ax.transAxes, fontsize=10, wrap=True)
                for column in range(len(cameras)+1, columns):
                    axes[row_index, column].set_axis_off()
            fig.suptitle(partition.replace('_', ' ').title() + '\nPost-hoc illustrative examples', fontsize=14)
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=.65, pad=.015,
                         label='Mean attention (shared log scale)')
            name = f'localization_examples_{partition}'
            paths.extend(_export(fig, output_dir, name))
            caption = ('Post-hoc examples selected from the highest-accuracy layer group in this '
                       'partition; ties choose the earliest group. These are not an independent '
                       'validation set. Mean heads, generated phrase tokens and the stated five '
                       'layers before raw global argmax over 384 positions. White bbox denotes '
                       'the target; + denotes the raw grid-cell center of the global peak, shown '
                       'on its actual camera. Heatmaps use one logarithmic normalization per '
                       'montage, mask zeros, and interpolate only for display. Missing image '
                       'panels do not change measured results.\n')
            _sidecars(output_dir, name, caption, metadata)
    return paths


_FIGURE = re.compile(r'^(causal_layer_(1|7|13|19|25)|localization_accuracy|localization_examples_(with|without)_coordinates)\.(png|pdf)$')
_EXTRA = re.compile(r'(caption|summary|summaries|audit|coverage|metadata|missing_images|image_fetch_receipt|selected_examples)', re.I)


def bundle_figures(output_root, figure_paths, metadata, extra_paths=()) -> Path:
    """Archive an explicit allowlist of figures and small text/numeric records."""
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    entries = []
    for figure, paths in ((True, figure_paths), (False, extra_paths)):
        for item in paths:
            path = Path(item).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Bundle members must be within output_root')
            if figure:
                valid = _FIGURE.fullmatch(path.name)
            else:
                valid = path.suffix.lower() in ('.json', '.jsonl', '.csv', '.txt', '.md') and _EXTRA.search(path.stem)
            if not valid:
                raise ValueError(f'File is outside the figure bundle allowlist: {path.name}')
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.name != 'metadata.json':
                entries.append(path)
    output = root / 'smolvlm_probing_figures.zip'
    temporary = output.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in dict.fromkeys(entries):
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr('metadata.json', json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise OSError('Figure ZIP failed CRC verification')
    temporary.replace(output)
    return output
