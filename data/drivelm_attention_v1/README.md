# DriveLM attention datasets — metadata freeze v1

**Status: metadata_frozen_not_visually_verified.** No model was run; no images were downloaded or visually inspected during filtering.

Input: 16,817 test questions, 189 six-camera frames, 35 scenes. Only the pinned prepared test split is eligible.
Reference answers and character spans are annotations, not generated answers/token indices.

Bản chốt theo metadata: chỉ test hiện tại, không chạy model và không duyệt ảnh. Hai tập localization tách theo tọa độ có sẵn trong câu hỏi; các reference span phải được căn lại với output model khi làm thí nghiệm.

| Main dataset | Questions | Targets | Frames | Scenes |
|---|---:|---:|---:|---:|
| causal | 2581 | 0 | 189 | 35 |
| localization_with_coordinates | 749 | 749 | 188 | 35 |
| localization_without_coordinates | 210 | 652 | 187 | 35 |

## What was selected

Concrete visual appearance, object inventories, spatial/lane relations, traffic signals/sign meanings, and specific scene/visibility descriptions.
Yes/no families, vehicle-state/motion questions, generic planning/probability answers, pure IDs, and structured bbox outputs are excluded.
A short sign meaning such as “No entry” is meaningful; an incidental “moving car” in a scene description is not a state question.
No answer-length ceiling, category/camera quota, or target sample count is imposed.

## Diversity and dependence

Repeated source Q/A and same-frame same-family/scope/information are merged. Different frames/objects are retained, even with the same answer.
Localization keeps at most one question per (coordinate partition, frame, object). One question can have several targets.
Seed 42 resolves ties, independent of model outputs. Repeated questions sharing images are not independent visual situations.
Per-scene/frame/category/camera/family/length/size counts and cross-task overlaps are in statistics.json; distributions reflect the eligible source, not a balanced benchmark.

## Annotation evidence and limitations

- Source annotation proves the object tag, camera, category, visual description, and bbox. Pixel geometry is 1600x900; it is not visually checked here.
- Reference bindings use a single explicitly queried object or a unique complete annotation description (case/punctuation/articles normalized). No nearest-box guessing or detached-list ordering is used.
- This conservative full-description matcher misses synonyms, paraphrases and unannotated objects. Missing bindings are not proof that an object is absent.
- Reference coverage is partial or unassessed. Ambiguous bindings are excluded; bbox validity is structural, not a claim of correct segmentation or exact image grounding.
- Unresolved entries distinguish source annotations with generic descriptions from actual reference spans; the former do not assert that the answer mentions those objects. Unmatched natural language is not exhaustively classified.
- Coordinate-guided and non-coordinate questions are separate. Non-coordinate questions can still specify a camera direction.
- Small bboxes are retained and labeled; an 8x8 visual grid can limit localization precision.
- No segmentation masks are created. A generated answer may omit or hallucinate an object; future experiments must record this instead of silently replacing samples.
- Only key objects are annotated. Reference spans must be aligned anew to actual generated tokens when experiments are designed.
- Do not feed reference answers, bbox annotations or reference spans to the model as extra prompt content. Original source questions, including their existing tags, are preserved.

## Files and reproducibility

The three JSONL files are the main datasets. audit.jsonl covers every source ID for all three outputs, with exclusions/merges/representatives; it is not a reserve dataset.
policy.json enumerates every observed question template, its family, rule and count. manifest.json pins input revisions/hashes, policy, seed, filtering code and output hashes.
Run the filtering command with the same metadata and seed to verify this freeze; incompatible sources/policy/code require a new output version.

From the Visualize repository root:

```powershell
python scripts/filter_drivelm_attention_data.py --metadata-dir .cache/drivelm-data-audit --output data/drivelm_attention_v1 --seed 42
```

Missing pinned metadata is downloaded automatically; existing files must pass their SHA-256 checks. Images and original large JSON files remain outside this bundle.

Sources: https://huggingface.co/datasets/minhdang0901/drivelm-nuscenes-v1-0 ; https://huggingface.co/datasets/huyvanzzz/prepared_data ; https://github.com/OpenDriveLab/DriveLM/blob/main/docs/data_prep_nus.md

## causal: selection audit

| Status | Questions |
|---|---:|
| excluded | 14236 |
| selected | 2581 |

| Main excluded/merged reason | Questions |
|---|---:|
| binary_question | 5455 |
| generic_planning_prediction_question | 4660 |
| state_motion_question | 3409 |
| tag_priority_question | 189 |
| ego_behavior_question | 189 |
| generic_or_empty_answer | 140 |
| structured_bbox_question | 109 |
| conflicting_reference_answers | 70 |
| generic_scene_comment | 9 |
| no_concrete_visual_content | 3 |
| wrong_type_appearance_answer | 2 |
| motion_only_answer | 1 |

## localization_with_coordinates: selection audit

| Status | Questions |
|---|---:|
| excluded | 15979 |
| merged | 89 |
| selected | 749 |

| Main excluded/merged reason | Questions |
|---|---:|
| binary_question | 5455 |
| generic_planning_prediction_question | 4660 |
| state_motion_question | 3409 |
| other_coordinate_partition | 1662 |
| tag_priority_question | 189 |
| ego_behavior_question | 189 |
| generic_or_empty_answer | 140 |
| structured_bbox_question | 109 |
| repeated_frame_object | 89 |
| no_provable_natural_reference_target | 81 |
| conflicting_reference_answers | 70 |
| generic_scene_comment | 9 |
| no_concrete_visual_content | 3 |
| wrong_type_appearance_answer | 2 |
| motion_only_answer | 1 |

## localization_without_coordinates: selection audit

| Status | Questions |
|---|---:|
| excluded | 16583 |
| merged | 24 |
| selected | 210 |

| Main excluded/merged reason | Questions |
|---|---:|
| binary_question | 5455 |
| generic_planning_prediction_question | 4660 |
| state_motion_question | 3409 |
| no_provable_natural_reference_target | 1428 |
| other_coordinate_partition | 919 |
| tag_priority_question | 189 |
| ego_behavior_question | 189 |
| generic_or_empty_answer | 140 |
| structured_bbox_question | 109 |
| conflicting_reference_answers | 70 |
| repeated_frame_object | 24 |
| generic_scene_comment | 9 |
| no_concrete_visual_content | 3 |
| wrong_type_appearance_answer | 2 |
| motion_only_answer | 1 |
