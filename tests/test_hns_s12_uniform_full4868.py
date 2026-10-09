from recovery import hns_s12_uniform_full4868 as r


def test_targets_and_lineage():
    assert r.TARGETS == (2434, 3651, 4868)
    assert r.predecessor(2434)[1] == 1217
    assert r.predecessor(3651)[1] == 2434
    assert r.predecessor(4868)[1] == 3651


def test_uniform_config_guard():
    cfg = r.protocol.common.read(r.BASE_EXP / 'config.json')
    r.frozen(cfg)
    assert cfg['view_sparsity_weights'] == [5 / 3, 5 / 3, 5 / 3]
