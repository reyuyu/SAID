import math
from recovery.s02_local_full import classify,expected_lrs,digest


def test_full_classification_boundaries():
    assert classify(72.768147,80)=='EARLY_SIGNAL_DID_NOT_SCALE'
    assert classify(72.768148,76.870244)=='FULL_POSITIVE'
    assert classify(73,76.870243)=='SHORT_LONG_TRADEOFF'


def test_continuation_lr_retains_native_horizon_and_group_order():
    from train.train_nested_semantic_mask import optimizer_learning_rates
    from model.balanced_hparam_search import BalancedSearch
    module=object.__new__(BalancedSearch)
    module.search_hparams=dict(visual_mask_lr_scale=1,fusion_lr=.0002)
    for completed in (500,504,1217,2434,3651,4867):
        assert expected_lrs(completed,module.search_hparams)==list(optimizer_learning_rates(module,completed,4868))
    assert expected_lrs(500,module.search_hparams)!=expected_lrs(0,module.search_hparams)


def test_stream_digest_detects_order_and_text_changes():
    assert digest({'sample_ids':[1,2],'views':['A','B']})!=digest({'sample_ids':[2,1],'views':['A','B']})
    assert digest({'tokens':[1,2]})!=digest({'tokens':[1,3]})


def test_dataset_survives_spawn_pickle():
    import pickle
    from recovery.s02_local_full import LocalDataset,INDEX,IMAGES
    data=LocalDataset(INDEX,IMAGES,'summary_random_detail',0)
    restored=pickle.loads(pickle.dumps(data))
    assert type(restored) is LocalDataset and restored.image_root==IMAGES
