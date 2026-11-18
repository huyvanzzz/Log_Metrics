"""Freeze annotation-only DriveLM test subsets, without running a model or reading images."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
import unicodedata
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.drivelm_localization import find_targets, normalize_tag

POLICY_VERSION = 'drivelm-attention-metadata-v1'
VALIDATION_STATUS = 'metadata_frozen_not_visually_verified'
SOURCE_REVISION = '6c4336a5279eaba3f3dfc5b525cc6ba57d55d61a'
PREPARED_REVISION = 'd2a6914f681dbb9148c4ac11e6fa92452010506e'
CAMERA_MAP = dict(CAM_FRONT='Front',CAM_FRONT_LEFT='Front-Left',CAM_FRONT_RIGHT='Front-Right',
                  CAM_BACK='Back',CAM_BACK_LEFT='Back-Left',CAM_BACK_RIGHT='Back-Right')
CAMERA_ORDER = list(CAMERA_MAP.values())
DATASET_NAMES = ('causal','localization_with_coordinates','localization_without_coordinates')
TAG = re.compile(r'<c\d+,[^<>]+>')
FILES = {
    'original_qa.json': ('minhdang0901/drivelm-nuscenes-v1-0', SOURCE_REVISION,
        'QA_dataset_nus/v1_0_train_nus.json','e7348866761a76ca1595646f410615ddd5adffb7affd7e056a203da0bf85a8cd'),
    'original_test.json': ('minhdang0901/drivelm-nuscenes-v1-0', SOURCE_REVISION,
        'multi_frame/multi_frame_test.json','7ef5fbf8a390b4b4206cb62138358cd5179119f091fa9780d1fa513e886f546c'),
    'prepared_test.jsonl': ('huyvanzzz/prepared_data', PREPARED_REVISION,
        'test.jsonl','0459b3120bcacd35044613ae89c653d01671344ae5bff28e701bd160cfc5c112'),
    'prepared_manifest.json': ('huyvanzzz/prepared_data', PREPARED_REVISION,
        'manifest.json','054a1cefdf6f1d8b2a9b7ef2c26d843f928f3e770fef42e3b579e6221d6ef36d'),
}


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def normalize(text):
    lookup=TAG.sub(lambda match:normalize_tag(match.group()),unicodedata.normalize('NFKC',text))
    return re.sub(r'\s+',' ',lookup.casefold()).strip().rstrip(' .!?;')


def question_template(question):
    return normalize(TAG.sub('<OBJECT>',question))


# These are all reviewed non-binary visual template families in the pinned test.
# The complete observed catalog, including exclusions, is exported in policy.json.
VISUAL_QUESTIONS = {
    'What is the visual description of <OBJECT>?':'object_appearance',
    'What is the traffic signal that the ego vehicle should pay attention to?':'traffic_signal',
    'What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.':'key_object_inventory',
    'Please describe the current scene.':'scene_description',
    'What is the relative positioning of the important objects in the current scene?':'spatial_relation',
    'Which lanes are each important object on in the scene?':'lane_relation',
    'What kind of traffic sign is <OBJECT>?':'traffic_sign_class',
    'What does <OBJECT> mean?':'traffic_sign_meaning',
    "What's your comment on this scene?":'scene_description',
    'What will affect driving judgment in this scene?':'visibility_environment',
    'What would affect driving judgment in this scene?':'visibility_environment',
    'What is worth noting about this scene?':'scene_description',
    'What is the visual description of this scene?':'scene_description',
    'What should be paid attention to in the current environment?':'visibility_environment',
    'What affects judgment?':'visibility_environment',
    'What should be noticed in this scene?':'scene_description',
}
for _direction in ('front','front left','front right','back','back left','back right'):
    VISUAL_QUESTIONS[f'What are objects to the {_direction} of the ego car?']='camera_object_inventory'
VISUAL_QUESTIONS = {question_template(key):value for key,value in VISUAL_QUESTIONS.items()}


def classify_question(question):
    text=question_template(question)
    def rule(family,decision,rule_id):
        return dict(family=family,decision=decision,rule_id=rule_id)
    if text in VISUAL_QUESTIONS:
        return rule(VISUAL_QUESTIONS[text],'candidate','reviewed_visual_template')
    if re.match(r'^(is|are|would|will|can|does|do|should|has|have)\b',text):
        return rule('binary_confirmation','excluded','binary_question')
    if text.startswith(('what is the status of','what is the observed status of object',
                         'what is the moving status of object','what is the future state of')):
        return rule('state_motion','excluded','state_motion_question')
    if text=='predict the behavior of the ego vehicle':
        return rule('state_motion','excluded','ego_behavior_question')
    if text.startswith('identify all the traffic elements in the front view, categorize them'):
        return rule('structured_bbox','excluded','structured_bbox_question')
    if text.startswith('what is the priority of the objects that the ego vehicle should consider'):
        return rule('tag_priority','excluded','tag_priority_question')
    prefixes=(
        'what object would consider <object> to be most relevant to its decision',
        'except for the ego vehicle, what object would consider <object> to be most relevant to its decision',
        'in this scenario, what object is most likely to consider <object>',
        'what actions taken by the ego vehicle can lead to a collision with <object>',
        'what actions could the ego vehicle take based on <object>',
        'which object is most likely to be occluded by <object>',
        'based on the observations of <object>, what are possible actions to be taken by <object>',
        'based on the observation of <object>, what actions may <object> take',
        'in this scenario, what are safe actions to take for the ego vehicle',
        'in this scenario, what are dangerous actions to take for the ego vehicle',
        'what is the target action of the ego vehicle',
        'based on <object> in this scene, what is the most possible action of the ego vehicle',
        'what object should the ego vehicle notice first when the ego vehicle is getting to the next possible location',
        'what is the probability of colliding with <object> after the ego vehicle',
    )
    if text.startswith(prefixes):
        return rule('planning_prediction','excluded','generic_planning_prediction_question')
    return rule('unreviewed','excluded','unreviewed_question_template')


GENERIC_ANSWERS = {
    '', 'yes','no','none','unknown','uncertain','low','medium','high','n/a','not sure','nothing',
    'worth noting','unable to determine','unable to confirm','cannot see clearly','unable to see clearly',
    'cannot determine','not known','no information','attention','guide',
}
GENERIC_SCENES = {
    "Watch out for the car's sudden acceleration.",
    'Please pay attention to starting with temporary parking.',
    'Complex road conditions.', 'Be careful in the dark for safety.',
    'This scene has a lot of information, the road conditions are complex, and it is worth paying extra attention.',
    'The scene has rich information, complex road conditions, and deserves special attention.',
    'This scene has a lot of information, the road conditions are complex, and it deserves special attention.',
    'The scene is very informative, with complex road conditions, and deserves special attention.',
    'This scene has a lot of information, the road conditions are complex, and it deserves special attention.',
    'The scene has a lot of information, the road conditions are complex, and it deserves special attention.',
}
GENERIC_SCENES = {normalize(value) for value in GENERIC_SCENES}
VISUAL_WORDS = set('car cars sedan sedans suv suvs truck trucks van vans pickup bus buses vehicle vehicles '
    'pedestrian pedestrians person people man woman child children cyclist cyclists bicycle bicycles bike bikes '
    'motorcycle motorcycles motorbike rider riders trailer trailers barrier barriers cone cones obstacle obstacles '
    'construction ambulance taxi taxis crosswalk sign signs light lights red green yellow white black blue gray grey '
    'silver brown orange pink purple helmet shirt clothes clothing luggage umbrella striped dotted road roads lane lanes '
    'raindrops droplets glass water mist fog rain rainy visibility visible dark night slippery congested intersection '
    'crossroad crossroads traffic pavement building buildings tree trees sidewalk sidewalks fence fences railing '
    'billboard arrow arrows pole poles parking parked empty narrow wide mirror mirrors'.split())
SIGNAL_MEANINGS = {'slow down and proceed','go straight','turn right','speed limit 40','slightly right straight'}


def exclusion_reason(record, classification):
    if classification['decision']!='candidate':
        return classification['rule_id']
    answer=normalize(record['answer'])
    if answer in GENERIC_ANSWERS or re.fullmatch(r'(?:very )?(?:low|medium|high)(?: probability| risk)?',answer):
        return 'generic_or_empty_answer'
    if answer in GENERIC_SCENES:
        return 'generic_scene_comment'
    if re.fullmatch(r'(?:vehicle|car|truck|bus) (?:skidding|moving|stationary|stopping)',answer):
        return 'motion_only_answer'
    # Coordinate tags remain in source text but cannot alone qualify an answer.
    natural=normalize(TAG.sub('',record['answer']))
    tokens=re.findall(r"[a-z]+(?:'[a-z]+)?|\d+",natural)
    if not tokens or not set(tokens)-{'and','the','ids','id','of','these','objects','are','is','object'}:
        return 'tag_or_coordinate_only_answer'
    family=classification['family']
    if family=='traffic_signal' and answer in SIGNAL_MEANINGS:
        return None
    if family in ('traffic_sign_meaning','traffic_sign_class'):
        return None
    if family in ('spatial_relation','lane_relation') and TAG.search(record['answer']):
        if re.search(r'\b(front|back|behind|left|right|ahead|lane|opposite|between|near)\b',natural):
            return None
    if family=='object_appearance' and answer in {'slow down','slow down to brake','guide'}:
        return 'wrong_type_appearance_answer'
    if family=='object_appearance' and answer=='the road is relatively empty':
        return 'wrong_type_appearance_answer'
    if family=='object_appearance' and answer in {'no entry','left turn or right turn'}:
        return None
    if not (set(tokens)&VISUAL_WORDS):
        return 'no_concrete_visual_content'
    return None


def _tie(seed,*values):
    return hashlib.sha256(canonical([seed,*values])).hexdigest()


def _scope(record):
    tags=sorted({normalize_tag(tag) for tag in TAG.findall(record['question'])})
    if tags:
        return ('objects',tuple(tags))
    question=record['question'].casefold()
    directions=re.findall(r'\b(?:front|back)(?: (?:left|right))?\b',question)
    return ('cameras',tuple(sorted(set(directions)))) if directions else ('frame',)


def _decision(status,reason,**kwargs):
    return dict(status=status,reason=reason,**kwargs)


def _common_record(row,classification,objects):
    result=copy.deepcopy(row)
    qt,at=TAG.findall(row['question']),TAG.findall(row['answer'])
    by_tag={normalize_tag(obj['object_tag']):obj for obj in objects}
    result.update(question_family=classification['family'],has_question_coordinates=bool(qt),
        has_answer_coordinates=bool(at),answer_word_count=len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+",TAG.sub('OBJECT',row['answer']))),
        metadata_object_references=[{key:by_tag[t][key] for key in ('object_tag','camera','category')}
            for t in dict.fromkeys(normalize_tag(tag) for tag in qt+at) if t in by_tag],
        validation_status=VALIDATION_STATUS)
    return result


def filter_records(records, objects_by_frame, seed=42):
    """Return three main datasets, exhaustive decisions and the reviewed template catalog."""
    rows=copy.deepcopy(list(records))
    ids=[r['sample_id'] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('Source sample IDs must be unique')
    by_id={r['sample_id']:r for r in rows}
    audit={r['sample_id']:dict(sample_id=r['sample_id'],eval_id=r['eval_id'],frame_id=r['frame_id'],
        scene_id=r['scene_id'],tasks={}) for r in rows}
    catalog={}
    for row in rows:
        template=question_template(row['question'])
        classification=classify_question(row['question'])
        entry=catalog.setdefault(template,dict(template=template,**classification,count=0))
        entry['count']+=1
    groups=defaultdict(list)
    for row in rows:
        groups[(row['frame_id'],normalize(row['question']))].append(row)
    representatives=[]
    aliases={}
    for group in groups.values():
        if len({normalize(row['answer']) for row in group})>1:
            for row in group:
                for name in DATASET_NAMES:
                    audit[row['sample_id']]['tasks'][name]=_decision('excluded','conflicting_reference_answers')
            continue
        first=min(group,key=lambda row:_tie(seed,row['sample_id']))
        representatives.append(first)
        aliases[first['sample_id']]=[r['sample_id'] for r in group if r['sample_id']!=first['sample_id']]
    eligible=[]
    local_candidates={name:[] for name in DATASET_NAMES[1:]}
    for row in representatives:
        sid=row['sample_id']
        classification=classify_question(row['question'])
        reason=exclusion_reason(row,classification)
        if reason:
            for name in DATASET_NAMES:
                audit[sid]['tasks'][name]=_decision('excluded',reason)
            continue
        objects=objects_by_frame.get(row['frame_id'],[])
        common=_common_record(row,classification,objects)
        common['source_alias_ids']=sorted(aliases[sid])
        eligible.append(common)
        targets,unresolved=find_targets(common,objects)
        name='localization_with_coordinates' if common['has_question_coordinates'] else 'localization_without_coordinates'
        other=next(n for n in DATASET_NAMES[1:] if n!=name)
        audit[sid]['tasks'][other]=_decision('excluded','other_coordinate_partition')
        if targets:
            common=copy.deepcopy(common)
            common.update(targets=targets,unresolved_reference=unresolved,reference_coverage='partial_or_unassessed',
                reference_span_kind='reference_characters_not_generated_token_indices')
            local_candidates[name].append(common)
        else:
            audit[sid]['tasks'][name]=_decision('excluded','no_provable_natural_reference_target',unresolved_reference=unresolved)
    # Semantic repetition is restricted to one frame/scope; never deduplicate across scenes/frames.
    semantic=defaultdict(list)
    for row in eligible:
        semantic[(row['frame_id'],row['question_family'],_scope(row),normalize(row['answer']))].append(row)
    datasets={name:[] for name in DATASET_NAMES}
    for group in semantic.values():
        first=min(group,key=lambda row:_tie(seed,row['sample_id']))
        datasets['causal'].append(first)
        audit[first['sample_id']]['tasks']['causal']=_decision('selected','concrete_visual_answer')
        for row in group:
            if row['sample_id']!=first['sample_id']:
                audit[row['sample_id']]['tasks']['causal']=_decision('merged','same_frame_semantic_repetition',representative_id=first['sample_id'])
    for name,candidates in local_candidates.items():
        instances=defaultdict(list)
        for row in candidates:
            for target in row['targets']:
                instances[(row['frame_id'],target['object_tag'])].append((row,target))
        winners={}
        for key,options in instances.items():
            winners[key]=min(options,key=lambda item:(item[1]['binding_rank'],_tie(seed,item[0]['sample_id'],item[1]['object_tag'])))[0]['sample_id']
        for row in candidates:
            selected=[]
            target_decisions=[]
            for target in row['targets']:
                winner=winners[(row['frame_id'],target['object_tag'])]
                chosen=winner==row['sample_id']
                if chosen:
                    selected.append(target)
                target_decisions.append(dict(object_tag=target['object_tag'],status='selected' if chosen else 'merged',representative_id=winner))
            sid=row['sample_id']
            if selected:
                item=copy.deepcopy(row);item['targets']=selected
                datasets[name].append(item)
                audit[sid]['tasks'][name]=_decision('selected','provable_reference_bbox_binding',targets=target_decisions,
                    unresolved_reference=row['unresolved_reference'])
            else:
                audit[sid]['tasks'][name]=_decision('merged','repeated_frame_object',targets=target_decisions,
                    representative_ids=sorted({t['representative_id'] for t in target_decisions}))
    for representative,losers in aliases.items():
        for loser in losers:
            for name in DATASET_NAMES:
                decision=audit[representative]['tasks'][name]
                if decision['status']=='excluded':
                    audit[loser]['tasks'][name]=copy.deepcopy(decision)
                else:
                    extras={'source_duplicate_of':representative}
                    if decision['status']=='selected':
                        extras['representative_id']=representative
                    elif 'representative_id' in decision:
                        extras['representative_id']=decision['representative_id']
                    else:
                        extras['representative_ids']=decision['representative_ids']
                    if 'targets' in decision:
                        extras['targets']=copy.deepcopy(decision['targets'])
                    audit[loser]['tasks'][name]=_decision('merged','duplicate_source_question_answer',**extras)
    for dataset in datasets.values():
        dataset.sort(key=lambda row:(row['eval_id'],row['sample_id']))
    result=dict(datasets=datasets,audit=[audit[r['sample_id']] for r in rows],
        template_rules=sorted(catalog.values(),key=lambda c:c['template']),seed=seed)
    validate_result(result,by_id)
    return result


def validate_result(result,source_by_id):
    if len(result['audit'])!=len(source_by_id) or {a['sample_id'] for a in result['audit']}!=set(source_by_id):
        raise ValueError('Audit does not cover the source test exactly')
    for name,rows in result['datasets'].items():
        selected={row['sample_id'] for row in rows}
        if len(selected)!=len(rows):
            raise ValueError('Duplicate selected sample')
        audited={a['sample_id'] for a in result['audit'] if a['tasks'][name]['status']=='selected'}
        if selected!=audited:
            raise ValueError('Dataset and audit selections differ')
        for a in result['audit']:
            decision=a['tasks'][name]
            if decision['status']=='merged':
                reps=decision.get('representative_ids',[])+([decision['representative_id']] if 'representative_id' in decision else [])
                if not reps or not set(reps)<=selected:
                    raise ValueError('Merge does not reference final selected samples')
        targets=set()
        for row in rows:
            source=source_by_id[row['sample_id']]
            if any(row[k]!=source[k] for k in ('sample_id','eval_id','question','answer','camera_paths')):
                raise ValueError('Source input was rewritten')
            if list(row['camera_paths'])!=CAMERA_ORDER:
                raise ValueError('Camera order changed')
            if exclusion_reason(row,classify_question(row['question'])):
                raise ValueError('Ineligible sample selected')
            if name!='causal':
                if not row.get('targets') or row['has_question_coordinates']!=(name=='localization_with_coordinates'):
                    raise ValueError('Invalid localization partition')
                for target in row['targets']:
                    key=(row['frame_id'],target['object_tag'])
                    if key in targets:
                        raise ValueError('Repeated localization instance')
                    targets.add(key)
                    span=target['reference_span']
                    if not 0<=span['start']<span['end']<=len(row['answer']) or row['answer'][span['start']:span['end']]!=span['text']:
                        raise ValueError('Invalid reference character span')


def _frame_key(paths):
    return tuple((camera,Path(path.replace('\\','/')).name) for camera,path in sorted(paths.items()))


def load_sources(directory):
    """Validate pinned inputs and recover frame/object annotations for all test records."""
    directory=Path(directory)
    provenance={}
    for local,(repo,revision,name,expected) in FILES.items():
        path=directory/local
        if not path.is_file():
            path.parent.mkdir(parents=True,exist_ok=True)
            url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
            temporary=path.with_suffix(path.suffix+'.download')
            print(f'Download metadata: {name}',flush=True)
            try:
                with urlopen(url,timeout=60) as response,temporary.open('wb') as stream:
                    shutil.copyfileobj(response,stream,1024*1024)
                os.replace(temporary,path)
            finally:
                temporary.unlink(missing_ok=True)
        actual=sha256(path)
        if actual!=expected:
            raise ValueError(f'Source SHA-256 mismatch: {local}')
        provenance[local]=dict(repository=repo,revision=revision,file=name,sha256=actual)
    prepared_manifest=json.loads((directory/'prepared_manifest.json').read_text(encoding='utf-8'))
    if prepared_manifest['counts']!={'train':341381,'val':19785,'test':16817}:
        raise ValueError('Unexpected prepared split counts')
    raw=json.loads((directory/'original_qa.json').read_text(encoding='utf-8'))
    test=json.loads((directory/'original_test.json').read_text(encoding='utf-8'))
    prepared=[json.loads(line) for line in (directory/'prepared_test.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(test)!=16817 or len(prepared)!=16817:
        raise ValueError('Expected complete 16,817-question test')
    frames={}
    for scene_id,scene in raw.items():
        for frame_id,frame in scene['key_frames'].items():
            key=_frame_key(frame['image_paths'])
            if key in frames or set(frame['image_paths'])!=set(CAMERA_MAP):
                raise ValueError('Ambiguous frame or missing source camera')
            frames[key]=(scene_id,frame_id,frame)
    if len(raw)!=696 or len(frames)!=4072:
        raise ValueError('Unexpected source scene/frame coverage')
    records=[];objects_by_frame={}
    for i,((qa,paths),p) in enumerate(zip(test,prepared)):
        if (p['sample_id']!=f'test-{i}' or p['eval_id']!=i or p['split']!='test' or
                p['question']!=qa['Q'] or p['answer']!=qa['A'] or list(p['camera_paths'])!=CAMERA_ORDER):
            raise ValueError(f'Prepared/source record mismatch at {i}')
        if set(paths)!=set(CAMERA_MAP) or any(paths[c]!=p['camera_paths'][CAMERA_MAP[c]] for c in CAMERA_MAP):
            raise ValueError(f'Camera path mismatch at {i}')
        match=frames.get(_frame_key(paths))
        if match is None:
            raise ValueError(f'Test frame missing at {i}')
        scene_id,frame_id,frame=match
        kind=qa['question_type']
        if not any(item['Q']==qa['Q'] and item['A']==qa['A'] for item in frame['QA'][kind]):
            raise ValueError(f'Q/A absent from original frame at {i}')
        objects=[]
        for object_tag,info in frame['key_object_infos'].items():
            parts=object_tag.strip('<>').split(',')
            if len(parts)!=4 or parts[1] not in CAMERA_MAP:
                raise ValueError('Malformed object tag')
            bbox=info['2d_bbox']
            if (len(bbox)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in bbox)
                    or not (0<=bbox[0]<bbox[2]<=1600 and 0<=bbox[1]<bbox[3]<=900)):
                raise ValueError('Invalid bbox annotation')
            objects.append(dict(object_tag=object_tag,camera=CAMERA_MAP[parts[1]],raw_camera=parts[1],
                bbox=bbox,category=info['Category'],description=info['Visual_description'],status=info['Status']))
        objects_by_frame[frame_id]=objects
        if any(normalize_tag(t) not in {normalize_tag(o['object_tag']) for o in objects} for t in TAG.findall(qa['Q']+' '+qa['A'])):
            raise ValueError('Unresolved source object tag')
        records.append(dict(**p,scene_id=scene_id,frame_id=frame_id,question_type=kind))
    if len(objects_by_frame)!=189 or len({r['scene_id'] for r in records})!=35:
        raise ValueError('Unexpected test frame/scene coverage')
    return records,objects_by_frame,provenance


def summarize(result):
    summary={}
    for name,rows in result['datasets'].items():
        lengths=sorted(row['answer_word_count'] for row in rows)
        references=[ref for row in rows for ref in row['metadata_object_references']]
        targets=[target for row in rows for target in row.get('targets',[])]
        summary[name]=dict(questions=len(rows),targets=len(targets),frames=len({row['frame_id'] for row in rows}),
            scenes=len({row['scene_id'] for row in rows}),question_family=dict(Counter(row['question_family'] for row in rows)),
            question_type=dict(Counter(row['question_type'] for row in rows)),
            questions_per_frame=dict(Counter(row['frame_id'] for row in rows)),
            questions_per_scene=dict(Counter(row['scene_id'] for row in rows)),
            question_coordinates=dict(Counter(str(row['has_question_coordinates']).lower() for row in rows)),
            answer_word_lengths=dict(min=min(lengths) if lengths else None,median=statistics.median(lengths) if lengths else None,
                max=max(lengths) if lengths else None,up_to5=sum(n<=5 for n in lengths),up_to10=sum(n<=10 for n in lengths),over20=sum(n>20 for n in lengths)),
            annotation_camera_references=dict(Counter(ref['camera'] for ref in references)),
            annotation_category_references=dict(Counter(ref['category'] for ref in references)),
            target_camera=dict(Counter(t['camera'] for t in targets)),target_category=dict(Counter(t['category'] for t in targets)),
            target_size=dict(Counter(t['size_bin'] for t in targets)),binding_methods=dict(Counter(t['binding_method'] for t in targets)),
            audit_status=dict(Counter(a['tasks'][name]['status'] for a in result['audit'])),
            exclusion_reasons=dict(Counter(a['tasks'][name]['reason'] for a in result['audit'] if a['tasks'][name]['status']!='selected')))
    sets={name:{row['sample_id'] for row in rows} for name,rows in result['datasets'].items()}
    summary['overlap_questions']={f'{a}__{b}':len(sets[a]&sets[b]) for i,a in enumerate(DATASET_NAMES) for b in DATASET_NAMES[i+1:]}
    return summary


def _json(value):
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'


def _jsonl(rows):
    return ''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n' for row in rows)


def _report(summary,source_count,seed):
    lines=['# DriveLM attention datasets — metadata freeze v1','',
        '**Status: metadata_frozen_not_visually_verified.** No model was run; no images were downloaded or visually inspected during filtering.','',
        f'Input: {source_count:,} test questions, 189 six-camera frames, 35 scenes. Only the pinned prepared test split is eligible.',
        'Reference answers and character spans are annotations, not generated answers/token indices.','',
        'Bản chốt theo metadata: chỉ test hiện tại, không chạy model và không duyệt ảnh. Hai tập localization tách theo tọa độ có sẵn trong câu hỏi; các reference span phải được căn lại với output model khi làm thí nghiệm.','',
        '| Main dataset | Questions | Targets | Frames | Scenes |', '|---|---:|---:|---:|---:|']
    for name in DATASET_NAMES:
        s=summary[name];lines.append(f"| {name} | {s['questions']} | {s['targets']} | {s['frames']} | {s['scenes']} |")
    lines+=['','## What was selected','',
        'Concrete visual appearance, object inventories, spatial/lane relations, traffic signals/sign meanings, and specific scene/visibility descriptions.',
        'Yes/no families, vehicle-state/motion questions, generic planning/probability answers, pure IDs, and structured bbox outputs are excluded.',
        'A short sign meaning such as “No entry” is meaningful; an incidental “moving car” in a scene description is not a state question.',
        'No answer-length ceiling, category/camera quota, or target sample count is imposed.','',
        '## Diversity and dependence','',
        'Repeated source Q/A and same-frame same-family/scope/information are merged. Different frames/objects are retained, even with the same answer.',
        'Localization keeps at most one question per (coordinate partition, frame, object). One question can have several targets.',
        f'Seed {seed} resolves ties, independent of model outputs. Repeated questions sharing images are not independent visual situations.',
        'Per-scene/frame/category/camera/family/length/size counts and cross-task overlaps are in statistics.json; distributions reflect the eligible source, not a balanced benchmark.','',
        '## Annotation evidence and limitations','',
        '- Source annotation proves the object tag, camera, category, visual description, and bbox. Pixel geometry is 1600x900; it is not visually checked here.',
        '- Reference bindings use a single explicitly queried object or a unique complete annotation description (case/punctuation/articles normalized). No nearest-box guessing or detached-list ordering is used.',
        '- This conservative full-description matcher misses synonyms, paraphrases and unannotated objects. Missing bindings are not proof that an object is absent.',
        '- Reference coverage is partial or unassessed. Ambiguous bindings are excluded; bbox validity is structural, not a claim of correct segmentation or exact image grounding.',
        '- Unresolved entries distinguish source annotations with generic descriptions from actual reference spans; the former do not assert that the answer mentions those objects. Unmatched natural language is not exhaustively classified.',
        '- Coordinate-guided and non-coordinate questions are separate. Non-coordinate questions can still specify a camera direction.',
        '- Small bboxes are retained and labeled; an 8x8 visual grid can limit localization precision.',
        '- No segmentation masks are created. A generated answer may omit or hallucinate an object; future experiments must record this instead of silently replacing samples.',
        '- Only key objects are annotated. Reference spans must be aligned anew to actual generated tokens when experiments are designed.',
        '- Do not feed reference answers, bbox annotations or reference spans to the model as extra prompt content. Original source questions, including their existing tags, are preserved.','',
        '## Files and reproducibility','',
        'The three JSONL files are the main datasets. audit.jsonl covers every source ID for all three outputs, with exclusions/merges/representatives; it is not a reserve dataset.',
        'policy.json enumerates every observed question template, its family, rule and count. manifest.json pins input revisions/hashes, policy, seed, filtering code and output hashes.',
        'Run the filtering command with the same metadata and seed to verify this freeze; incompatible sources/policy/code require a new output version.','',
        'From the Visualize repository root:','',
        '```powershell',
        'python scripts/filter_drivelm_attention_data.py --metadata-dir .cache/drivelm-data-audit --output data/drivelm_attention_v1 --seed 42',
        '```','',
        'Missing pinned metadata is downloaded automatically; existing files must pass their SHA-256 checks. Images and original large JSON files remain outside this bundle.','',
        'Sources: https://huggingface.co/datasets/minhdang0901/drivelm-nuscenes-v1-0 ; https://huggingface.co/datasets/huyvanzzz/prepared_data ; https://github.com/OpenDriveLab/DriveLM/blob/main/docs/data_prep_nus.md','']
    for name in DATASET_NAMES:
        s=summary[name];lines+= [f'## {name}: selection audit','', '| Status | Questions |','|---|---:|']
        lines += [f'| {status} | {count} |' for status,count in sorted(s['audit_status'].items())]
        lines+=['','| Main excluded/merged reason | Questions |','|---|---:|']
        lines += [f'| {reason} | {count} |' for reason,count in sorted(s['exclusion_reasons'].items(),key=lambda item:-item[1])]
        lines+=['']
    return '\n'.join(lines)


def write_bundle(result,output_dir,provenance,seed=42):
    output=Path(output_dir)
    summary=summarize(result)
    policy=dict(version=POLICY_VERSION,seed=seed,validation_status=VALIDATION_STATUS,
        state_motion_questions='excluded',selection='maximum_nonredundant_metadata_eligible',
        camera_order=CAMERA_ORDER,answer_length_ceiling=None,visual_review=False,
        normalization='NFKC casefold whitespace collapse and trailing sentence punctuation; source text unchanged',
        localization_binding='single_query_tag_attribute or unique_complete_annotation_description',
        source_duplicate_conflicts='exclude entire normalized frame/question group',
        template_rules=result['template_rules'])
    payloads={f'{name}.jsonl':_jsonl(rows) for name,rows in result['datasets'].items()}
    payloads.update({'audit.jsonl':_jsonl(result['audit']),'policy.json':_json(policy),
                     'statistics.json':_json(summary),'README.md':_report(summary,len(result['audit']),seed)})
    hashes={name:hashlib.sha256(content.encode('utf-8')).hexdigest() for name,content in payloads.items()}
    identity=dict(policy_version=POLICY_VERSION,seed=seed,sources=provenance,
        code_sha256={name:sha256(Path(__file__).parent/name) for name in ('filter_drivelm_attention_data.py','drivelm_localization.py')},
        policy_sha256=hashes['policy.json'],file_sha256=hashes)
    manifest=dict(version=1,validation_status=VALIDATION_STATUS,identity=identity,
        fingerprint=hashlib.sha256(canonical(identity)).hexdigest(),source_questions=len(result['audit']),
        counts={name:dict(questions=summary[name]['questions'],targets=summary[name]['targets'],frames=summary[name]['frames'],scenes=summary[name]['scenes']) for name in DATASET_NAMES})
    payloads['manifest.json']=_json(manifest)
    if output.exists():
        if not output.is_dir() or {p.name for p in output.iterdir()}!=set(payloads):
            raise ValueError('Output already exists with different files; choose a new version directory')
        if any((output/name).read_bytes()!=content.encode('utf-8') for name,content in payloads.items()):
            raise ValueError('Output already exists with different source/policy/code; choose a new version directory')
        print(f'Existing metadata freeze verified: {output}',flush=True)
        return manifest
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+'.',dir=output.parent))
    try:
        for name,content in payloads.items():
            (stage/name).write_bytes(content.encode('utf-8'))
        os.replace(stage,output)
    finally:
        if stage.exists():
            if stage.resolve().parent != output.parent.resolve() or not stage.name.startswith(output.name+'.'):
                raise RuntimeError('Refusing cleanup outside the output staging directory')
            shutil.rmtree(stage)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir',type=Path,default=ROOT/'.cache/drivelm-data-audit')
    parser.add_argument('--output',type=Path,default=ROOT/'data/drivelm_attention_v1')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    records,objects,provenance=load_sources(args.metadata_dir)
    result=filter_records(records,objects,args.seed)
    unknown=[rule for rule in result['template_rules'] if rule['family']=='unreviewed']
    if unknown:
        raise ValueError(f'Question templates require explicit policy review before freeze: {unknown}')
    manifest=write_bundle(result,args.output,provenance,args.seed)
    print(json.dumps(dict(output=str(args.output),fingerprint=manifest['fingerprint'],counts=manifest['counts']),indent=2))


if __name__=='__main__':
    main()
