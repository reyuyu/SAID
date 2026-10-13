"""CPU checks for fixed caption handling. No model is loaded."""
from extract_long_dci_features import first_sentence, make_caption_versions


def main():
    cases = [
        ('Dr. Smith stands outside. A tree is visible.', 'Dr. Smith stands outside.'),
        ('A U.S. flag waves. The sky is blue.', 'A U.S. flag waves.'),
        ('A 3.5 inch object is visible. Another object follows.', 'A 3.5 inch object is visible.'),
        ('A sign says "Stop." Another car is parked.', 'A sign says "Stop."'),
        ('A BIG LOTS! store. There is a car.', 'A BIG LOTS! store.'),
        ('What is visible? A tree.', 'What is visible?'),
    ]
    for source, expected in cases:
        value, meta = first_sentence(source)
        assert value == expected, (source, value, expected)
        assert meta['boundary_found']
    sentence, meta = first_sentence('No punctuation is present')
    assert sentence == 'No punctuation is present' and meta['fallback_full_caption']
    ids, stats = make_caption_versions('green tree ' * 300)
    assert stats['full_kept_bpe_tokens'] == 246
    assert stats['prefix50_kept_bpe_tokens'] == 123
    assert ids['full'].shape == ids['prefix50'].shape == (248,)
    assert ids['full'][1:124].equal(ids['prefix50'][1:124])
    print('Caption CPU regression checks: PASS')


if __name__ == '__main__':
    main()
