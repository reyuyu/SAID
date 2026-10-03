# Official Urban1K reproduction status

**Not executed: official CE and BCE checkpoint downloads failed.**
Source reference is the [pinned official README](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/README.md).
These values are published references, not our measured results:

| Checkpoint | README T2I % | README I2T % | Measured official CLS | Measured official TCI |
|---|---:|---:|---|---|
| CE, beta0.5,K36 |89.0|88.6|unavailable|unavailable|
| BCE, beta0.5,K36 |91.8|92.3|unavailable|unavailable|

Correct direction mapping for future SAID-format reporting is I2T/T2I88.6/89.0
for CE and92.3/91.8 for BCE. Their actual checkpoint reproduction and internal
CLS/TCI attribution are unknown. No protocol adjustment is made to match them.

No model was reconstructed, so strict-load keys, native embedding max errors,
full1000-pool ranking equality and official-versus-harness recall equality have
not been validated. Five-dataset comparison remains stopped at this gate.
See commands/*console.txt for actual failed-download evidence.
