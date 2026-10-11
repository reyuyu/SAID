# DCI protocol investigation

Status: **PROTOCOL_UNVERIFIED**.

All five public HyFL commits inspected. Initial/latest evaluation blob hashes match; middle commits only alter README. The only evaluation file reads filename/caption from absent DCI_test.json, sorts filenames and gives no generation rule. Paper Table1 separates DCI and Long-DCI; README ambiguously names dci Long-DCI and links TULIP.

Inspected author GitHub repositories, UNCHA evaluation/config, author HF dataset listings (empty), model repository files (weights and README only), HF DCI search and public issues (empty). None provides a HyFL-author annotation or complete rule. This is bounded public-source evidence, not a claim to have searched every private/unindexed resource.

GOAL pinned commit `fd3ea90b63f9c7fd8c3231018a57bd0ad84ac888` has 1999 records, all in SAID, but lacks 5806 of 7,805. Only 1 of 1999 shared captions is token-identical to short+extra. Its candidate set explicitly mismatches; no GPU evaluation or substitution was performed.

SAID raw BPE mean excluding SOT/EOT is 172.187444; including them approximately 174.187. The proximity to paper 174.2 is auxiliary, not identity proof. 1,383 captions exceed 246 content tokens before truncation.

Current DCI remains the explicit short+space+extra/strip convention. No alternate caption version was scored. Author annotation or exact constructor still required; request text prepared but never sent.
