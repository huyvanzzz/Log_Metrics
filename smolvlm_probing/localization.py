"""Literal generated-phrase binding and auditable global-peak localization.

These CPU-only helpers never tokenize a response again or infer semantic object
identity. Maps must already be averaged across heads, phrase tokens, and layers
within each group by the attention collector.
"""
import math
import re
from collections import Counter


CAMERAS = ("Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right")
LAYER_GROUPS = tuple((start, start + 4) for start in range(0, 30, 5))
PARTITIONS = ("with_coordinates", "without_coordinates")


def _as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _decision(target, status, reason, span=None, indices=()):
    return {"object_tag": target.get("object_tag"), "status": status,
            "reason": reason, "token_indices": list(indices),
            "matched_span": span, "target": target}


def match_targets(tokenizer, generated_ids, targets, *, generation_status="valid"):
    """Bind each frozen literal phrase to original pre-EOS generated indices.

    ``generated_ids`` must retain every pre-EOS token, including specials in the
    middle. Prefix instability excludes otherwise unique matches rather than
    falling back to re-encoding, offsets from another tokenization, or guessing.
    """
    targets = list(targets)
    ids = _as_list(generated_ids)
    if generation_status not in {"valid", "length", "eos", "empty"}:
        return [_decision(t, "invalid_generation", str(generation_status)) for t in targets]
    specials = set(getattr(tokenizer, "all_special_ids", ()) or ())
    if generation_status == "empty" or not any(i not in specials for i in ids):
        return [_decision(t, "empty_generation", "no_content_tokens") for t in targets]

    def decode(values):
        return tokenizer.decode(values, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)

    try:
        response = decode(ids)
    except Exception as exc:
        return [_decision(t, "unalignable_tokens", "response_decode_failed: " + type(exc).__name__)
                for t in targets]
    if not response:
        return [_decision(t, "empty_generation", "empty_decoded_response") for t in targets]

    decisions = []
    for target in targets:
        phrase = target["reference_span"]["text"]
        # Lookahead also counts overlapping repeats such as "a a" in "a a a".
        matches = list(re.finditer(r"(?=(?<!\w)(" + re.escape(phrase) + r")(?!\w))", response)) if phrase else []
        if not matches:
            decisions.append(_decision(target, "missing_phrase", "literal_reference_phrase_not_found"))
        elif len(matches) > 1:
            decisions.append(_decision(target, "repeated_phrase", "literal_reference_phrase_not_unique"))
        else:
            m = matches[0]
            span = {"start": m.start(1), "end": m.end(1), "text": m.group(1)}
            decisions.append(_decision(target, "eligible", "unique_literal_reference_phrase", span))

    # Decide conflicts together; never resolve them by target order or geometry.
    ambiguous = set()
    candidates = [(i, d) for i, d in enumerate(decisions) if d["status"] == "eligible"]
    for n, (i, left) in enumerate(candidates):
        for j, right in candidates[n + 1:]:
            a, b = left["matched_span"], right["matched_span"]
            if a["start"] < b["end"] and b["start"] < a["end"]:
                ambiguous.update((i, j))
    for i in ambiguous:
        decisions[i].update(status="ambiguous_binding", reason="overlapping_target_bindings")

    if not any(d["status"] == "eligible" for d in decisions):
        return decisions
    intervals = []
    alignment_reason = None
    try:
        previous = decode([])
        if previous != "":
            alignment_reason = "nonempty_empty_prefix"
        for i, token in enumerate(ids):
            prefix = decode(ids[:i + 1])
            if not response.startswith(prefix) or len(prefix) < len(previous):
                alignment_reason = "unstable_prefix_decoding"
                break
            if token not in specials and len(prefix) > len(previous):
                intervals.append((i, len(previous), len(prefix)))
            previous = prefix
    except Exception as exc:
        alignment_reason = "prefix_decode_failed: " + type(exc).__name__
    for decision in decisions:
        if decision["status"] != "eligible":
            continue
        span = decision["matched_span"]
        indices = [i for i, start, end in intervals if start < span["end"] and end > span["start"]]
        if alignment_reason or not indices:
            decision.update(status="unalignable_tokens", reason=alignment_reason or "no_regular_overlapping_tokens")
        else:
            decision["token_indices"] = indices
    return decisions


def _geometry(target):
    width, height = target["image_size"]
    bbox = target["bbox"]
    if (len(bbox) != 4 or not all(math.isfinite(float(v)) for v in (*bbox, width, height))
            or width <= 0 or height <= 0):
        raise ValueError("invalid target image size or bbox")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
        raise ValueError("bbox must be ordered and inside the image")
    if target["camera"] not in CAMERAS:
        raise ValueError("unknown target camera")
    return float(width), float(height), bbox


def _center(index, width, height):
    row, col = divmod(index % 64, 8)
    return [(col + .5) * width / 8, (row + .5) * height / 8]


def _inside(center, bbox):
    x, y = center
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_contains_grid_center(target):
    """Whether any 8x8 cell center falls inside the target's inclusive bbox."""
    width, height, bbox = _geometry(target)
    return any(_inside(_center(i, width, height), bbox) for i in range(64))


def evaluate_targets(decisions, target_maps, *, layer_groups=LAYER_GROUPS):
    """Evaluate averaged [eligible targets, groups, 384] maps in eligible order.

    A single global argmax determines camera and cell. Ties pick the lowest
    index, including zero maps, and do not change eligibility or denominators.
    Numerical or shape errors raise rather than silently excluding a target.
    """
    decisions = list(decisions)
    maps = [] if target_maps is None else _as_list(target_maps)
    groups = tuple(tuple(group) for group in layer_groups)
    eligible_count = sum(d["status"] == "eligible" for d in decisions)
    if len(maps) != eligible_count:
        raise ValueError("target_maps must contain exactly one map per eligible target")
    results, map_index = [], 0
    for decision in decisions:
        target = decision["target"]
        width, height, bbox = _geometry(target)
        reachable = bbox_contains_grid_center(target)
        result = {**decision, "bbox_contains_grid_center": reachable, "group_results": []}
        if decision["status"] == "eligible":
            target_map = _as_list(maps[map_index])
            map_index += 1
            if len(target_map) != len(groups):
                raise ValueError("target map group count does not match layer_groups")
            for (start, end), values in zip(groups, target_map):
                values = [float(value) for value in _as_list(values)]
                if len(values) != 384 or not all(math.isfinite(value) for value in values):
                    raise ValueError("each localization map must have 384 finite values")
                maximum = max(values)
                peak = values.index(maximum)
                camera = CAMERAS[peak // 64]
                center = _center(peak, width, height)
                result["group_results"].append({
                    "layer_start": start, "layer_end": end, "peak_index": peak,
                    "peak_camera": camera, "peak_center": center,
                    "correct": camera == target["camera"] and _inside(center, bbox),
                    "bbox_contains_grid_center": reachable,
                    "tied_peak_count": values.count(maximum),
                    "zero_map": all(value == 0 for value in values)})
        results.append(result)
    return results


def aggregate_localization(records):
    """Summarize target audits by the two frozen coordinate partitions.

    Each record is {sample_id, partition, targets: evaluated_audits}. Coverage
    and accuracy without a denominator are None, serialized as JSON null.
    ``reasons`` counts final status labels, including eligible, so its counts
    sum to ``total_targets``. Duplicate sample records are rejected.
    """
    by_partition = {partition: [] for partition in PARTITIONS}
    seen = set()
    for record in records:
        partition = record["partition"]
        if partition not in by_partition:
            raise ValueError("unknown localization partition: " + str(partition))
        identity = (partition, record["sample_id"])
        if identity in seen:
            raise ValueError("duplicate localization sample: " + str(identity))
        seen.add(identity)
        by_partition[partition].extend(record["targets"])
    summary = {}
    for partition, targets in by_partition.items():
        eligible = [target for target in targets if target["status"] == "eligible"]
        groups = []
        for index, (start, end) in enumerate(LAYER_GROUPS):
            correct = 0
            for target in eligible:
                results = target["group_results"]
                if len(results) != len(LAYER_GROUPS):
                    raise ValueError("eligible target must have all six layer groups")
                group = results[index]
                if (group["layer_start"], group["layer_end"]) != (start, end):
                    raise ValueError("localization layer groups are inconsistent")
                correct += bool(group["correct"])
            groups.append({"layers": [start, end], "correct": correct,
                           "total": len(eligible),
                           "accuracy": correct / len(eligible) if eligible else None})
        summary[partition] = {
            "total_targets": len(targets), "eligible_targets": len(eligible),
            "coverage": len(eligible) / len(targets) if targets else None,
            "reasons": dict(sorted(Counter(target["status"] for target in targets).items())),
            "groups": groups,
            "unreachable_total": sum(not target["bbox_contains_grid_center"] for target in targets),
            "unreachable_eligible": sum(not target["bbox_contains_grid_center"] for target in eligible)}
    return summary
