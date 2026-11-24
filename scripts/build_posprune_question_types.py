"""Index the annotated test questions against the pinned prepared test file."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'smolvlm_attention/posprune_ablation/test_question_types.json'


def build(annotated_path, prepared_path):
    annotated_bytes, prepared_bytes = annotated_path.read_bytes(), prepared_path.read_bytes()
    annotated = json.loads(annotated_bytes)
    prepared = [json.loads(line) for line in prepared_bytes.splitlines() if line.strip()]
    labels = []
    for index, ((qa, cameras), row) in enumerate(zip(annotated, prepared, strict=True)):
        expected_cameras = {f"CAM_{name.upper().replace('-', '_')}": path
                            for name, path in row['camera_paths'].items()}
        if (row['eval_id'] != index or row['question'] != qa['Q'] or row['answer'] != qa['A']
                or expected_cameras != cameras):
            raise ValueError(f'Annotated and prepared test rows differ at index {index}')
        label = qa['question_type']
        if not isinstance(label, str) or not label:
            raise ValueError(f'Missing or invalid question type at index {index}')
        labels.append(label)

    counts = Counter(labels)
    types = sorted(counts)
    if not types or len(types) > 10:
        raise ValueError('Expected up to ten named question types')
    codes = {name: str(i) for i, name in enumerate(types)}
    return dict(prepared_sha256=sha256(prepared_bytes).hexdigest(),
                annotation_sha256=sha256(annotated_bytes).hexdigest(),
                question_types=types, counts=dict(sorted(counts.items())),
                labels_by_eval_id=''.join(codes[name] for name in labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotated_test', type=Path)
    parser.add_argument('prepared_test', type=Path)
    args = parser.parse_args()
    value = build(args.annotated_test, args.prepared_test)
    OUTPUT.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
