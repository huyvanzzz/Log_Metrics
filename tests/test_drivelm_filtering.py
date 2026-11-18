"""Selection contracts: annotation-driven, source-preserving, and frozen."""
import copy
import json
from pathlib import Path

import pytest

from scripts import filter_drivelm_attention_data as filtering


CAMERAS = ['Front','Front-Left','Front-Right','Back','Back-Left','Back-Right']
TAG = '<c1,CAM_FRONT,100.0,100.0>'


def record(index, question, answer, frame='frame-a'):
    return dict(sample_id=f'test-{index}',eval_id=index,split='test',scene_id='scene-a',frame_id=frame,
                question=question,answer=answer,
                camera_paths={camera:f'data/nuscenes/samples/{camera}/{frame}-{camera}.jpg' for camera in CAMERAS},
                question_type='perception')


def object_info(tag=TAG, description='White sedan.'):
    return dict(object_tag=tag,camera='Front',raw_camera='CAM_FRONT',bbox=[50.,50.,150.,150.],
                category='Vehicle',description=description,status='Moving')


@pytest.mark.parametrize('question,answer',[
    ('Are there cars in front of the ego vehicle?','Yes, a white car is ahead.'),
    ('What is the visual description of '+TAG+'?','Unknown.'),
    ('What is the observed status of object '+TAG+'?','Moving.'),
    ('What is the moving status of object '+TAG+'?','Turn left.'),
    ('What is the future state of '+TAG+'?','Keep going straight.'),
    ('Predict the behavior of the ego vehicle.','The ego vehicle is moving slowly.'),
    ('What is the traffic signal that the ego vehicle should pay attention to?','None.'),
    ('What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.',TAG),
])
def test_unwanted_rows_excluded(question,answer):
    result=filtering.filter_records([record(0,question,answer)],{'frame-a':[object_info()]})
    assert not result['datasets']['causal']
    assert result['audit'][0]['tasks']['causal']['status']=='excluded'


def test_short_specific_and_incidental_motion_are_kept():
    rows=[record(0,'What is the traffic signal that the ego vehicle should pay attention to?','Red light.'),
          record(1,'Please describe the current scene.','There is a moving white sedan in front of the ego car.'),
          record(2,'What does '+TAG+' mean?','No entry.')]
    original=copy.deepcopy(rows)
    result=filtering.filter_records(rows,{'frame-a':[object_info()]})
    assert len(result['datasets']['causal'])==3
    assert rows==original
    for output in result['datasets']['causal']:
        source=rows[output['eval_id']]
        assert all(output[k]==source[k] for k in ('sample_id','eval_id','question','answer','camera_paths'))
        assert list(output['camera_paths'])==CAMERAS


def test_duplicates_conflicts_and_cross_frame_retention():
    q='What is the visual description of '+TAG+'?'
    rows=[record(0,q,'White sedan.'),record(1,q,'White sedan.'),record(2,q,'White sedan.',frame='frame-b'),
          record(3,'Please describe the current scene.','A pedestrian is ahead.'),
          record(4,'Please describe the current scene.','There is no pedestrian ahead.')]
    result=filtering.filter_records(rows,{'frame-a':[object_info()],'frame-b':[object_info()]})
    kept=result['datasets']['causal']
    assert len(kept)==2 and {r['frame_id'] for r in kept}=={'frame-a','frame-b'}
    audit={r['sample_id']:r for r in result['audit']}
    assert all(audit[f'test-{i}']['tasks']['causal']['reason']=='conflicting_reference_answers' for i in (3,4))
    merged=[a for a in result['audit'] if a['tasks']['causal']['status']=='merged']
    assert len(merged)==1 and merged[0]['tasks']['causal']['representative_id'] in {r['sample_id'] for r in kept}


def test_localization_partition_unique_objects_and_reference_spans():
    rows=[record(0,'What is the visual description of '+TAG+'?','White sedan.'),
          record(1,'Please describe the current scene.','A white sedan is in front of the ego car.'),
          record(2,'What are objects to the front of the ego car?','One white sedan is ahead.')]
    result=filtering.filter_records(rows,{'frame-a':[object_info()]})
    for name in ('localization_with_coordinates','localization_without_coordinates'):
        targets=[t for row in result['datasets'][name] for t in row['targets']]
        assert len(targets)==1
        target=targets[0]
        row=result['datasets'][name][0]
        span=target['reference_span']
        assert row['answer'][span['start']:span['end']]==span['text']
        assert target['object_tag']==TAG and target['visual_verified'] is False
        assert row['validation_status']=='metadata_frozen_not_visually_verified'


def test_ambiguous_instance_not_selected_and_small_bbox_retained():
    row=record(0,'Please describe the current scene.','A white sedan is ahead.')
    a=object_info();b=object_info('<c2,CAM_FRONT,200.0,100.0>')
    result=filtering.filter_records([row],{'frame-a':[a,b]})
    assert not result['datasets']['localization_without_coordinates']
    guided=record(1,'What is the visual description of '+TAG+'?','White sedan.')
    result=filtering.filter_records([guided],{'frame-a':[a,b]})
    assert result['datasets']['localization_with_coordinates'][0]['targets'][0]['size_bin']=='small'


def test_repeatable_hashes_and_existing_output_protection(tmp_path):
    rows=[record(0,'What is the visual description of '+TAG+'?','White sedan.')]
    result=filtering.filter_records(rows,{'frame-a':[object_info()]})
    provenance={'fixture':True}
    filtering.write_bundle(result,tmp_path/'v1',provenance,seed=42)
    first={p.name:p.read_bytes() for p in (tmp_path/'v1').iterdir()}
    filtering.write_bundle(result,tmp_path/'v1',provenance,seed=42)
    assert first=={p.name:p.read_bytes() for p in (tmp_path/'v1').iterdir()}
    manifest=json.loads((tmp_path/'v1/manifest.json').read_text())
    assert manifest['validation_status']=='metadata_frozen_not_visually_verified'
    with pytest.raises(ValueError,match='different|new version'):
        filtering.write_bundle(result,tmp_path/'v1',provenance,seed=43)
    assert first=={p.name:p.read_bytes() for p in (tmp_path/'v1').iterdir()}


def test_full_test_audit_accounting_and_no_model_dependencies():
    rows=[record(0,'What is the visual description of '+TAG+'?','White sedan.'),
          record(1,'Is there a car ahead?','Yes.')]
    result=filtering.filter_records(rows,{'frame-a':[object_info()]})
    assert len(result['audit'])==len(rows)
    for dataset in result['datasets']:
        statuses=[row['tasks'][dataset]['status'] for row in result['audit']]
        assert all(status in ('selected','excluded','merged') for status in statuses)
        assert statuses.count('selected')==len(result['datasets'][dataset])
    source=Path(filtering.__file__).read_text(encoding='utf-8')
    assert 'predictions.jsonl' not in source and 'import torch' not in source


def test_tag_spacing_is_lookup_only_source_text_is_preserved():
    spaced='<c1, CAM_FRONT, 100.0, 100.0>'
    a=record(0,'What is the visual description of '+TAG+'?','White sedan.')
    b=record(1,'What is the visual description of '+spaced+'?','White sedan.')
    result=filtering.filter_records([a,b],{'frame-a':[object_info()]})
    assert len(result['datasets']['causal'])==1
    item=result['datasets']['localization_with_coordinates'][0]
    assert item['question'] in (a['question'],b['question'])
    assert item['metadata_object_references'][0]['object_tag']==TAG


def test_reviewed_context_specific_answers_and_wrong_type_appearance():
    rows=[record(0,'What is the traffic signal that the ego vehicle should pay attention to?','Go straight.'),
          record(1,'What is the visual description of '+TAG+'?','Rearview mirror.'),
          record(2,'What is the visual description of '+TAG+'?','No entry.',frame='frame-b'),
          record(3,'What is the visual description of '+TAG+'?','The road is relatively empty.',frame='frame-c'),
          record(4,"What's your comment on this scene?",'The scene has a lot of information, the road conditions are complex, and it deserves special attention.')]
    # Different question families and frames avoid confounding source-conflict handling.
    rows[1]['frame_id']='frame-d'
    objects={frame:[object_info()] for frame in ('frame-a','frame-b','frame-c','frame-d')}
    result=filtering.filter_records(rows,objects)
    assert {r['sample_id'] for r in result['datasets']['causal']}=={'test-0','test-1','test-2'}


def test_merged_alias_points_to_final_selected_representative():
    q='What is the visual description of '+TAG+'?'
    for i in range(2,12):
        rows=[record(0,q,'White sedan.'),record(1,q,'White sedan.'),
              record(i,'What kind of traffic sign is '+TAG+'?','White sedan.')]
        result=filtering.filter_records(rows,{'frame-a':[object_info()]})
        for name,selected_rows in result['datasets'].items():
            selected={row['sample_id'] for row in selected_rows}
            for a in result['audit']:
                decision=a['tasks'][name]
                if decision['status']=='merged' and 'representative_id' in decision:
                    assert decision['representative_id'] in selected
