from recovery.hierarchy090_full_failure_audit import json_normalize, key_type_differences


def test_native_histogram_vs_JSON_receipt_key_type_boundary():
    native={'nested_d3_statistics':{'K_eff_histogram':{0:7,1:12,3:237}},'K_histogram_by_n':{'4':{'2':5}}}
    logged=json_normalize(native)
    assert native!=logged and json_normalize(native)==logged
    differences=key_type_differences(native,logged)
    assert len(differences)==3 and all(x['in_memory_key_type']=='int' for x in differences)
    changed={'nested_d3_statistics':{'K_eff_histogram':{'0':7,'1':12,'3':236}},'K_histogram_by_n':{'4':{'2':5}}}
    assert json_normalize(native)!=changed, 'Normalization must not erase real count drift'
