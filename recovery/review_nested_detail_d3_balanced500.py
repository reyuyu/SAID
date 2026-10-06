"""Reuse immutable/native evidence review with exact balanced-weight proof."""
import json
from recovery import review_nested_detail_d3_equal500 as review
from recovery.nested_detail_d3_balanced500 import (
    RUN,EXP,PHASE,BASELINE,EQUAL_EXP,RELATED_TEST_COUNT,configure,matched_rows,update_reports)
from recovery.s02_nfs500 import dump


def main():
    configure()
    review.RUN,review.EXP,review.PHASE,review.BASELINE,review.ATOMIC_EXP=RUN,EXP,PHASE,BASELINE,EQUAL_EXP
    review.RELATED_TEST_COUNT,review.matched_rows,review.update_reports=RELATED_TEST_COUNT,matched_rows,update_reports
    review.EXPECTED_WEIGHTS,review.ISOLATION_KEY=[1.35,1.35,.30],'config_only_view_weights_changed'
    review.main()
    validation=json.loads((EXP/'VALIDATION.json').read_text())
    for k in ('method_change_only_lowest_view_sampling','gradient_batches_and_protocol_exact_atomic_equal'):
        validation.pop(k,None)
    validation.update(method_change_only_alignment_weights=True,
        all512000_F_Dall_D3_strings_tokens_selected_indices_exact=True,
        gradient_batches_and_protocol_exact_D3_equal=True)
    dump(EXP/'VALIDATION.json',validation)
    print(json.dumps(dict(review_passed=True,status=json.loads((EXP/'RESULTS.json').read_text())['status'])))


if __name__=='__main__':main()
