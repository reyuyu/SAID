"""Reuse the completed native evidence review, then write equal-weight comparisons."""
import json
from recovery import review_nested_detail500 as protocol
from recovery.nested_detail_equal_weight500 import RUN,EXP,PHASE,BASELINE,WEIGHTS,BASELINE_EXP,configure,update_reports
from recovery.s02_nfs500 import dump


def main():
    configure()
    protocol.RUN,protocol.EXP,protocol.PHASE,protocol.BASELINE,protocol.WEIGHTS = RUN,EXP,PHASE,BASELINE,WEIGHTS
    protocol.review()
    result = json.loads((EXP/'RESULTS.json').read_text())
    assert result['launch_provenance']['alignment_weight_only_proof']['config_only_view_weights_changed']
    assert result['stream_proof']['all500_sample_ids_F_Dall_Ds_strings_tokens_exact']
    previous = json.loads((BASELINE_EXP/'GRADIENT_SPOTCHECK.json').read_text())
    current = json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    assert [b['global_sample_ids_sha256'] for b in current['batches']]==[b['global_sample_ids_sha256'] for b in previous['batches']]
    update_reports()
    validation = json.loads((EXP/'VALIDATION.json').read_text())
    validation.pop('related_unit_tests_passed',None)
    validation.update(alignment_weights_sole_method_change=True,
        related_unit_tests_passed=result['launch_provenance']['related_unit_tests']['passed'],
        all_three_view_strings_tokens_exact_prior_nested=True,
        gradient_batch_IDs_and_protocol_exact_prior_nested=True)
    dump(EXP/'VALIDATION.json',validation)
    print(json.dumps({'status':result['status'],'review_passed':True,'scores':result['scores_percent']}))


if __name__=='__main__':
    main()
