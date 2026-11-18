import unittest
from scripts import drivelm_localization as loc


def obj(n=1, description='White sedan.', camera='Front'):
    return dict(object_tag=f'<c{n},CAM_FRONT,{n},2>', camera=camera,
                raw_camera='CAM_FRONT', bbox=[0, 0, 160, 90],
                category='Vehicle', description=description, status='')


class LocalizationTests(unittest.TestCase):
    def find(self, answer, objects, question='What objects are in the front view?'):
        return loc.find_targets(dict(question=question, answer=answer), objects)

    def test_unique_description_and_exact_reference_offsets(self):
        a='There is a WHITE sedan ahead.'
        targets, unresolved=self.find(a, [obj()])
        self.assertEqual(len(targets), 1)
        t=targets[0]
        self.assertEqual(a[t['reference_span']['start']:t['reference_span']['end']], 'WHITE sedan')
        self.assertEqual(t['size_bin'], 'medium')
        self.assertFalse(t['visual_verified'])
        self.assertEqual(unresolved, [])

    def test_ambiguity_never_uses_list_order(self):
        targets, unresolved=self.find('A white sedan. <c1,CAM_FRONT,1,2> <c2,CAM_FRONT,2,2>', [obj(),obj(2)])
        self.assertEqual(targets, [])
        self.assertTrue(unresolved)

    def test_single_question_tag_attribute_binding(self):
        targets, _=self.find('White.', [obj()], 'What color is <c1,CAM_FRONT,1,2>?')
        self.assertEqual(targets[0]['reference_span']['text'], 'White')

    def test_tag_only_and_spatial_answers_are_insufficient(self):
        for a in ['<c1,CAM_FRONT,1,2>', 'To the left.']:
            self.assertEqual(self.find(a,[obj()])[0], [])

    def test_camera_scope_does_not_pick_nearest(self):
        targets,_=self.find('White sedan.', [obj(),obj(2,camera='Back')])
        self.assertEqual([t['object_tag'] for t in targets], [obj()['object_tag']])

    def test_partial_multi_object_keeps_unique_and_reports_missing(self):
        targets,unresolved=self.find('White sedan and blue truck <c9,CAM_FRONT,9,2>.', [obj()])
        self.assertEqual(len(targets),1)
        self.assertTrue(any(x['reason']=='missing_annotation' for x in unresolved))

    def test_longest_description_suppresses_nested_match(self):
        targets,_=self.find('White sedan.', [obj(),obj(2,'Sedan.')])
        self.assertEqual(len(targets),1)
        self.assertEqual(targets[0]['object_tag'],obj()['object_tag'])

    def test_plural_does_not_claim_single_instance(self):
        self.assertEqual(self.find('White sedans.', [obj()])[0], [])

    def test_explicit_local_tag_can_disambiguate_description(self):
        targets,_=self.find('White sedan (<c1,CAM_FRONT,1,2>).', [obj(),obj(2)])
        self.assertEqual([t['object_tag'] for t in targets],[obj()['object_tag']])

    def test_invalid_box_is_not_a_target(self):
        o=obj(); o['bbox']=[4,3,2,1]
        self.assertEqual(self.find('White sedan.', [o])[0], [])

    def test_attribute_span_excludes_coordinate_segment(self):
        targets,_=self.find('White sedan (120, 240).', [obj()], 'What does <c1,CAM_FRONT,1,2> look like?')
        self.assertEqual(targets[0]['reference_span']['text'], 'White sedan')

    def test_nonappearance_question_tag_does_not_bind_action(self):
        self.assertEqual(self.find('Turn right.',[obj()], 'What action should follow <c1,CAM_FRONT,1,2>?')[0], [])

    def test_negated_description_is_unresolved(self):
        for answer in ['There is no white sedan here.', 'Without a white sedan, we can move.']:
            targets,unresolved=self.find(answer,[obj()])
            self.assertEqual(targets, [])
            self.assertTrue(any(u['reason']=='negated_reference' for u in unresolved))

    def test_description_does_not_bridge_sentences(self):
        self.assertEqual(self.find('It is white. Sedan drivers wait.',[obj()])[0], [])

    def test_no_entry_is_description_not_precontext_negation(self):
        targets,_=self.find('A no entry sign.',[obj(description='No entry sign.')])
        self.assertEqual(len(targets),1)

    def test_question_tag_cannot_bypass_negated_object_mention(self):
        targets,unresolved=self.find('There is no white sedan here.', [obj()],
                                    'What is the visual description of <c1,CAM_FRONT,1,2>?')
        self.assertEqual(targets, [])
        self.assertTrue(any(u['reason']=='negated_reference' for u in unresolved))

    def test_question_tag_whitespace_lookup_preserves_annotation_tag(self):
        targets,unresolved=self.find('White.', [obj()], 'What color is <c1, CAM_FRONT, 1, 2>?')
        self.assertEqual([t['object_tag'] for t in targets],[obj()['object_tag']])
        self.assertEqual(unresolved,[])

    def test_adjacent_tag_whitespace_preserves_reference_offsets(self):
        answer='White sedan (<c1, CAM_FRONT, 1, 2>).' 
        targets,unresolved=self.find(answer,[obj(),obj(2)])
        self.assertEqual([t['object_tag'] for t in targets],[obj()['object_tag']])
        span=targets[0]['reference_span']
        self.assertEqual(answer[span['start']:span['end']], 'White sedan')
        self.assertEqual(unresolved,[])

    def test_uncertain_descriptions_rejected_independently_of_row(self):
        for phrase in ['Unknown', 'Unable to see clearly', 'Unable to confirm',
                       'Unable to determine', 'Cannot identify', 'Not known']:
            targets,unresolved=self.find('White sedan; '+phrase+'.', [obj(),obj(2,phrase)])
            self.assertEqual([t['object_tag'] for t in targets],[obj()['object_tag']])
            self.assertTrue(any(u['reason']=='generic_object_description' for u in unresolved))

    def test_question_tag_generic_attribute_rejected(self):
        for phrase in ['Unknown.', 'Unknown (120, 240).', 'Unable to confirm.', 'Cannot determine.']:
            targets,unresolved=self.find(phrase,[obj()], 'What color is <c1,CAM_FRONT,1,2>?')
            self.assertEqual(targets,[])
            self.assertTrue(any(u['reason']=='generic_object_description' for u in unresolved))
