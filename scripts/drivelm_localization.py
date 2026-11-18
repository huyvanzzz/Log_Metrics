"""Conservative reference-annotation bindings; never loads images or models.

The caller owns QA eligibility. Coordinates in source tags are identifiers only.
Description matching deliberately sacrifices recall for auditable uniqueness.
"""
import math
import re

TAG = re.compile(r'<c\d+,[^<>]+>')
WORD = re.compile(r"[A-Za-z0-9]+")
ARTICLES = {'a', 'an', 'the'}
SPATIAL = set('to the left right front back behind ahead above below in on at of is it there here object vehicle this that and'.split())


def normalize_tag(tag):
    """Normalize identifier whitespace without modifying source text or offsets."""
    return re.sub(r'\s+', '', tag)


def is_generic_description(description):
    """Recognize closed uncertainty answers, without rejecting real nouns."""
    value = ' '.join(m.group().lower() for m in WORD.finditer(str(description or '')))
    return (value in {'', 'unknown', 'unclear', 'uncertain', 'unspecified',
                      'unidentified', 'unidentifiable', 'none', 'null', 'n a',
                      'na', 'not sure', 'not visible', 'no information',
                      'not available', 'cannot tell', 'can t tell', 'i don t know'}
            or bool(re.match(r'^(?:unable\b|cannot\b|can not\b|can t\b|not known\b|not determined\b)', value)))


def _words(text):
    return [(m.group().lower(), m.start(), m.end()) for m in WORD.finditer(text)
            if m.group().lower() not in ARTICLES]


def _scope(question):
    cameras = set()
    for match in re.finditer(r'\b(front|back|rear)(?:[ -](left|right))?\b', TAG.sub('', question.lower())):
        side = 'Back' if match[1] in ('back', 'rear') else 'Front'
        cameras.add(side + ('-' + match[2].title() if match[2] else ''))
    return cameras if len(cameras) == 1 else set()


def _negated_before(answer, start):
    """Conservative clause-local negation, excluding the matched phrase itself."""
    context = re.split(r'[.!?;\n]|\b(?:but|however)\b', answer[:start], flags=re.I)[-1]
    return bool(re.search(r"\b(?:no|not|without|neither|never|absence of|lack of)\b|n't\b", context, re.I))


def _target(obj, answer, start, end, method, rank):
    bbox = obj.get('bbox')
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox):
        return None
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1600 and 0 <= y1 < y2 <= 900):
        return None
    fraction = (x2-x1)*(y2-y1)/(1600*900)
    return {**{key: obj.get(key, '') for key in
               ('object_tag', 'camera', 'raw_camera', 'category', 'description')},
            'bbox': list(bbox), 'image_size': [1600, 900],
            'reference_span': {'start': start, 'end': end, 'text': answer[start:end]},
            'binding_method': method, 'binding_rank': rank,
            'size_fraction': fraction,
            'size_bin': 'small' if fraction < .01 else 'medium' if fraction < .1 else 'large',
            'visual_verified': False}


def find_targets(record, objects):
    """Return (targets, unresolved) using only source QA and object metadata.

    Full descriptions match after case/punctuation/article normalization.
    An immediately parenthesized tag can disambiguate that description. Merely
    appending a tag list does not establish an ordinal description association.
    """
    question, answer = str(record.get('question', '')), str(record.get('answer', ''))
    by_tag = {normalize_tag(o['object_tag']): o for o in objects if o.get('object_tag')}
    unresolved = []
    generic_tags = {normalize_tag(o['object_tag']) for o in objects
                    if o.get('object_tag') and is_generic_description(o.get('description'))}
    for tag in sorted(generic_tags):
        unresolved.append({'object_tag': by_tag[tag]['object_tag'],
                           'reason': 'generic_object_description',
                           'evidence_scope': 'source_annotation_not_asserted_reference_mention'})
    for tag in dict.fromkeys(TAG.findall(question + ' ' + answer)):
        if normalize_tag(tag) not in by_tag:
            unresolved.append({'object_tag': tag, 'reason': 'missing_annotation'})
    # Replace tags with equal-length spaces to retain original character offsets.
    natural = TAG.sub(lambda m: ' ' * len(m.group()), answer)
    words = _words(natural)
    values = [w[0] for w in words]
    qtags = list(dict.fromkeys(normalize_tag(tag) for tag in TAG.findall(question)))
    appearance = re.search(r'visual description|look like|appearance|what (?:is the )?colou?r|what type|what kind', question, re.I)
    if len(qtags) == 1 and appearance and qtags[0] in by_tag and words:
        if qtags[0] in generic_tags:
            return [], unresolved
        if is_generic_description(natural):
            unresolved.append({'object_tag': by_tag[qtags[0]]['object_tag'],
                               'reason': 'generic_object_description',
                               'reference_span': {'start': words[0][1], 'end': words[-1][2],
                                                  'text': answer[words[0][1]:words[-1][2]]}})
            return [], unresolved
        # Attribute binding applies only to an actual natural answer, excluding
        # tag-only, numeric/coordinate-only and purely spatial references.
        # Select one contiguous natural segment. Tags and numeric coordinate
        # tuples cannot occur inside the evidence span.
        blocked = list(TAG.finditer(answer)) + list(re.finditer(
            r'[\[(]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?[^\])]*[\])]', answer))
        segments, cursor = [], 0
        for match in sorted(blocked, key=lambda m: m.start()):
            if match.start() >= cursor:
                segments.append((cursor, match.start()))
            cursor = max(cursor, match.end())
        segments.append((cursor, len(answer)))
        chunks = [[w for w in words if w[1] >= a and w[2] <= b] for a,b in segments]
        attribute_words = max(chunks, key=lambda ws: sum(w[0] not in SPATIAL and not w[0].isdigit() for w in ws), default=[])
        meaningful = [w[0] for w in attribute_words if w[0] not in SPATIAL and not w[0].isdigit()]
        description_words = [w[0] for w in _words(str(by_tag[qtags[0]].get('description', '')))]
        negated_description = bool(description_words) and any(
            values[i:i+len(description_words)] == description_words
            and _negated_before(answer, words[i][1])
            for i in range(len(values)-len(description_words)+1))
        if meaningful and not negated_description:
            start, end = attribute_words[0][1], attribute_words[-1][2]
            if is_generic_description(answer[start:end]):
                unresolved.append({'object_tag': by_tag[qtags[0]]['object_tag'],
                                   'reason': 'generic_object_description',
                                   'reference_span': {'start': start, 'end': end, 'text': answer[start:end]}})
                return [], unresolved
            target = _target(by_tag[qtags[0]], answer, start, end, 'question_tag_attribute', 0)
            if target:
                return [target], unresolved
    scope = _scope(question)
    eligible = [o for o in objects if not is_generic_description(o.get('description'))
                and (not scope or o.get('camera') in scope)]
    matches = {}
    for obj in eligible:
        desc = [w[0] for w in _words(str(obj.get('description', '')))]
        if not desc or not any(w not in SPATIAL and not w.isdigit() for w in desc):
            continue
        for i in range(len(values)-len(desc)+1):
            if values[i:i+len(desc)] != desc:
                continue
            start, end = words[i][1], words[i+len(desc)-1][2]
            # Never bridge a coordinate tag to synthesize a natural phrase.
            if TAG.search(answer[start:end]) or re.search(r'[.!?;\n]', answer[start:end]):
                continue
            matches.setdefault((start,end), []).append(obj)
    targets, covered, used = [], [], set()
    for (start, end), candidates in sorted(matches.items(), key=lambda item: (-(item[0][1]-item[0][0]), item[0][0])):
        if any(start >= a and end <= b for a,b in covered):
            continue
        covered.append((start,end))
        if _negated_before(answer, start):
            unresolved.append({'reason': 'negated_reference',
                               'reference_span': {'start': start, 'end': end, 'text': answer[start:end]}})
            continue
        adjacent = re.match(r'\s*\(\s*(<c\d+,[^<>]+>)\s*\)', answer[end:])
        method, rank = 'unique_description', 2
        if adjacent:
            candidates = [o for o in candidates if normalize_tag(o.get('object_tag', '')) == normalize_tag(adjacent[1])]
            method, rank = 'explicit_tagged_description', 1
        unique = {o.get('object_tag'): o for o in candidates}
        if len(unique) != 1:
            unresolved.append({'reason': 'ambiguous_description' if unique else 'description_tag_mismatch',
                               'reference_span': {'start': start, 'end': end, 'text': answer[start:end]},
                               'candidate_tags': sorted(unique)})
            continue
        obj = next(iter(unique.values()))
        if obj['object_tag'] in used:
            continue
        target = _target(obj, answer, start, end, method, rank)
        if target:
            targets.append(target)
            used.add(obj['object_tag'])
        else:
            unresolved.append({'reason': 'invalid_bbox', 'object_tag': obj['object_tag']})
    targets.sort(key=lambda t: t['reference_span']['start'])
    return targets, unresolved
