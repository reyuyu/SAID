# Actual Full Flickr30k evaluation

31,783 images; 158,915 captions; one complete candidate pool; explicit five positives per image.

I2T R@1/5/10: 56.712708 / 78.969890 / 85.866658%.

T2I R@1/5/10: 36.725293 / 59.293333 / 68.391908%.

Correct counts: `{"I2T": {"1": 18025, "5": 25099, "10": 27291}, "T2I": {"1": 58362, "5": 94226, "10": 108685}}`.

Exact ties use ascending fixed manifest index; near ties use unrounded FP32 cosine. Query256/gallery4096; no full score matrix.
