"""Resolve the trained predictor before downloading the experimental dataset."""

import os
from hashlib import file_digest
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from tarfile import open as open_tar
from tempfile import TemporaryDirectory, mkdtemp
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile


def _kaggle_reference(value):
    parsed = urlsplit(value)
    if parsed.scheme:
        parts = parsed.path.strip('/').split('/')
        if parsed.hostname not in ('www.kaggle.com', 'kaggle.com') or len(parts) < 3 or parts[0] != 'code':
            raise ValueError('Use a Kaggle notebook URL or owner/notebook-slug[/version]')
        version = parse_qs(parsed.query).get('scriptVersionId', [''])[0]
        if version:
            raise ValueError('For a specific Kaggle version, enter owner/notebook-slug/version using its version number')
        value = '/'.join(parts[1:3])
    if not re.fullmatch(r'[\w-]+/[\w-]+(?:/[1-9]\d*)?', value, re.ASCII):
        raise ValueError('Use a Kaggle notebook URL or owner/notebook-slug[/version]')
    return value


def _download_kaggle(directory, reference, filename, secret):
    reference = _kaggle_reference(reference)
    file = PurePosixPath(filename)
    if file.is_absolute() or '..' in file.parts or file.suffix.lower() not in ('.pt', '.zip'):
        raise ValueError('PREDICTOR_KAGGLE_FILE must be a relative .pt or .zip output filename')
    executable = shutil.which('kaggle')
    if executable is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], check=True)
        executable = shutil.which('kaggle')
    if executable is None:
        raise RuntimeError('The Kaggle CLI executable is unavailable')
    environment = dict(os.environ)
    if secret:
        from kaggle_secrets import UserSecretsClient

        environment['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret(secret)
    directory.mkdir(parents=True, exist_ok=True)
    destination = Path(mkdtemp(prefix='kaggle-output-', dir=directory))
    # Download the file through the CLI API without attaching a notebook input.
    prefix = r'(?:.*/)?' if len(file.parts) == 1 else ''
    subprocess.run([executable, 'kernels', 'output', reference, '--path', str(destination),
                    '--file-pattern', f'^{prefix}{re.escape(file.as_posix())}$'], env=environment, check=True)
    matches = [path for path in destination.rglob(file.name) if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f'Kaggle output {reference} must provide exactly one {filename}; found {len(matches)}')
    return matches[0]


def _download_kaggle_model(directory, reference, member, expected_sha256):
    import requests

    if not re.fullmatch(r'(?:[\w-]+/){4}[1-9]\d*', reference, re.ASCII):
        raise ValueError('Use the Kaggle Model handle owner/model/framework/variation/version')
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('A versioned Kaggle Model source requires the checkpoint SHA-256')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'{expected_sha256}.pt'
    if target.is_file():
        with target.open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() == expected_sha256:
                return target
    with TemporaryDirectory(prefix='kaggle-model-', dir=directory) as temporary:
        archive = Path(temporary) / 'model.tar.gz'
        url = f'https://www.kaggle.com/api/v1/models/{reference}/download'
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with archive.open('wb') as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
        candidate = Path(temporary) / 'predictor.pt'
        with open_tar(archive, 'r:gz') as bundle:
            entries = [entry for entry in bundle if entry.isfile() and entry.name == member]
            if len(entries) != 1:
                raise ValueError(f'Expected exactly one {member} in Kaggle Model {reference}; found {len(entries)}')
            with bundle.extractfile(entries[0]) as source, candidate.open('wb') as stream:
                shutil.copyfileobj(source, stream)
        with candidate.open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != expected_sha256:
                raise ValueError(f'Checkpoint SHA-256 mismatch for {member} in Kaggle Model {reference}')
        candidate.replace(target)
    return target


def resolve_predictor_input(directory, *, checkpoint='', hf_repo='', hf_revision='', hf_filename='predictor_best.pt',
                            hf_repo_type='model', kaggle_run='', kaggle_file='predictor_best.pt', kaggle_secret='',
                            kaggle_model='', kaggle_model_member='', kaggle_model_sha256=''):
    directory = Path(directory)
    if sum(bool(value) for value in (checkpoint, hf_repo, kaggle_run, kaggle_model)) != 1:
        raise ValueError('Set exactly one predictor source in the first cell: PREDICTOR_CHECKPOINT, '
                         'PREDICTOR_HF_REPO, PREDICTOR_KAGGLE_RUN, or PREDICTOR_KAGGLE_MODEL.')
    if kaggle_model:
        path = _download_kaggle_model(directory, kaggle_model, kaggle_model_member, kaggle_model_sha256)
    elif hf_repo:
        if not hf_revision:
            raise ValueError('Set PREDICTOR_HF_REVISION for the repository containing the trained checkpoint')
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=hf_repo, revision=hf_revision, filename=hf_filename, repo_type=hf_repo_type))
    elif kaggle_run:
        path = _download_kaggle(directory, kaggle_run, kaggle_file, kaggle_secret)
    else:
        path = Path(checkpoint).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f'The predictor checkpoint source is not a file: {path}')
    if path.suffix.lower() != '.zip':
        return path
    with ZipFile(path) as bundle:
        matches = [entry for entry in bundle.infolist()
                   if not entry.is_dir() and PurePosixPath(entry.filename).name == 'predictor_best.pt']
        if len(matches) != 1:
            raise ValueError(f'The predictor output ZIP must contain one predictor_best.pt; found {len(matches)}')
        directory.mkdir(parents=True, exist_ok=True)
        target = Path(mkdtemp(prefix='predictor-zip-', dir=directory)) / 'predictor_best.pt'
        with bundle.open(matches[0]) as source, target.open('wb') as stream:
            shutil.copyfileobj(source, stream)
    return target
