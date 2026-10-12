from train.train_nested_semantic_mask import component_ablation_cycle_policy


def test_three_second_boundary_is_warning_only():
    first = component_ablation_cycle_policy(3.1, 0)
    second = component_ablation_cycle_policy(3.2, first['consecutive'])
    third = component_ablation_cycle_policy(4.0, second['consecutive'])
    assert first['warning'] and second['warning'] and third['warning']
    assert third['consecutive'] == 3
    assert not third['stop']


def test_fast_cycle_resets_slow_counter():
    assert component_ablation_cycle_policy(2.9, 2) == {
        'warning': False, 'consecutive': 0, 'stop': False, 'reason': None}


def test_sixty_second_cycle_is_hard_stop():
    result = component_ablation_cycle_policy(60.0001, 99)
    assert result['warning'] and result['consecutive'] == 100
    assert result['stop']
    assert result['reason'] == 'single full update exceeded 60 seconds'
