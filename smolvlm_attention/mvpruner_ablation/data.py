"""Read the local DriveLM test split and resolve unambiguous camera references."""

from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile


CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
TEST_COUNT = 16_817
IMAGE_COUNT = 1_134
MAP_LAYERS = (2, 15, 30)

_NUMBER = r'-?\d+(?:\.\d+)?'
_TAG = re.compile(rf'<c\d+,\s*(CAM_[A-Z_]+),\s*({_NUMBER}),\s*({_NUMBER})\s*>', re.I)
_DIRECTION = r'(?:front|back)(?: (?:left|right))?'
_PROSE = re.compile(rf'\b(?:to the ({_DIRECTION}) of the ego car|(front) view)\b', re.I)


@contextmanager
def atomic_output(path):
    """Write beside the destination so an interrupted write leaves the prior file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.close()
            os.replace(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_output(path) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def file_sha256(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def camera_label(question):
    tags = [(camera.upper(), float(x), float(y)) for camera, x, y in _TAG.findall(question)]
    tagged = {camera for camera, _, _ in tags}
    prose = {'CAM_' + (relative or explicit).upper().replace(' ', '_')
             for relative, explicit in _PROSE.findall(_TAG.sub('', question))}
    cameras = tagged | prose
    if not cameras.issubset(CAMERAS):
        raise ValueError(f'Unknown camera in question: {question}')
    if len(cameras) != 1:
        return None, 'multiple' if cameras else 'none', tags
    return next(iter(cameras)), 'tag' if tagged else 'prose', tags


def _relative_image(value, camera):
    relative = PurePosixPath(value.replace('\\', '/'))
    if relative.parts[:1] == ('data',):
        relative = PurePosixPath(*relative.parts[1:])
    if (len(relative.parts) != 4 or relative.parts[:3] != ('nuscenes', 'samples', camera)
            or relative.suffix.lower() not in ('.jpg', '.jpeg', '.png')
            or not relative.stem or ':' in relative.name):
        raise ValueError(f'Invalid image path for {camera}: {value}')
    return relative.as_posix()


def prepare_manifest(test_file, image_root, destination, *, example_id=None):
    from PIL import Image

    payload = Path(test_file).read_bytes()
    records = json.loads(payload)
    if not isinstance(records, list) or len(records) != TEST_COUNT:
        raise ValueError(f'Expected all {TEST_COUNT:,} test questions')
    root = Path(image_root)
    frames, rows, counts = {}, [], Counter()
    image_paths = set()
    for index, (qa, views) in enumerate(records):
        question = qa['Q']
        if not isinstance(question, str) or not question.strip() or set(views) != set(CAMERAS):
            raise ValueError(f'Invalid question or camera set at test row {index}')
        paths = [_relative_image(views[camera], camera) for camera in CAMERAS]
        frame_id = sha256('\n'.join(paths).encode()).hexdigest()
        frames.setdefault(frame_id, paths)
        image_paths.update(paths)
        camera, source, tags = camera_label(question)
        counts[source] += 1
        rows.append(dict(id=index, frame=frame_id, question=question, camera=camera,
                         source=source, tags=tags))
    if len(image_paths) != IMAGE_COUNT:
        raise ValueError(f'Expected {IMAGE_COUNT:,} distinct images, found {len(image_paths):,}')

    assets = {}
    for relative in sorted(image_paths):
        path = root / relative
        with Image.open(path) as image:
            size = image.size
            image.verify()
        assets[relative] = dict(size=size, sha256=file_sha256(path))

    example = None
    candidates = rows if example_id is None else [row for row in rows if row['id'] == example_id]
    for row in candidates:
        if row['source'] != 'tag':
            continue
        camera = row['camera']
        relative = frames[row['frame']][CAMERAS.index(camera)]
        width, height = assets[relative]['size']
        for _, x, y in row['tags']:
            if 0 <= x < width and 0 <= y < height:
                example = dict(id=row['id'], camera=camera, x=x, y=y)
                break
        if example is not None:
            break
    if example is None:
        raise ValueError('No requested illustration question has an in-bounds object tag')

    for row in rows:
        del row['tags']
    label_counts = {source: dict(Counter(row['camera'] for row in rows if row['source'] == source))
                    for source in ('tag', 'prose')}
    manifest = dict(version=1, test_sha256=sha256(payload).hexdigest(), cameras=CAMERAS,
                    counts=dict(counts), label_counts=label_counts, frames=frames, rows=rows,
                    assets=assets, illustration=example, map_layers=MAP_LAYERS)
    write_json(destination, manifest)
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_bytes())
    if manifest['version'] != 1 or tuple(manifest['cameras']) != CAMERAS:
        raise ValueError('Unsupported ablation manifest')
    if len(manifest['rows']) != TEST_COUNT or len(manifest['assets']) != IMAGE_COUNT:
        raise ValueError('Manifest must cover the complete test split')
    return manifest


def rows_by_frame(manifest):
    groups = {frame_id: [] for frame_id in manifest['frames']}
    for row in manifest['rows']:
        groups[row['frame']].append(row)
    return groups
