# GitHub synchronization

Target repository: `reyuyu/SAID`.

The user requests that future completed code/report changes be committed and
pushed to GitHub, with the resulting commit link included in the handoff.

## Authorized recovery synchronization: 2026-10-05

The user explicitly requested a status update and GitHub synchronization.
The destination is the dedicated `recovery/sa1b-restoration` branch, based on
`52bb7ae`; historical experiment branches remain unchanged. The commit author is
the agent identity `Codex Recovery <codex-recovery@users.noreply.github.com>`,
not an impersonation of the repository owner. No global Git identity is changed.
The handoff reports the actual remote-confirmed commit after push; this document
does not substitute for that confirmation.

A dedicated Ed25519 deploy key has been generated on this server:

- Public key: `/root/.ssh/said_github_deploy_ed25519.pub`
- Private key: `/root/.ssh/said_github_deploy_ed25519`
- Fingerprint: `SHA256:CyK0DkgmXpMEU7wdH5yePqI8Y3XnnjjyJYrXSA6XiwU`

Add only the public key in the repository's Settings > Deploy keys and enable
Allow write access. Never upload, print, or commit the private key.

The public key was added by the user. SSH authentication returned the repository
identity `reyuyu/SAID`, and read access plus a write-access dry run succeeded.
The dry run did not create a branch or upload commits. Actual branch updates
remain subject to repository protections.

Origin is now `git@github.com:reyuyu/SAID.git`. Repository-local
`core.sshCommand` selects the dedicated private key, noninteractive authentication,
and strict host-key checking using `/root/.ssh/said_github_known_hosts`.
The observed GitHub Ed25519 host fingerprint was checked against the published
fingerprint `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`.

The earlier deploy-key setup performed no commit or actual push. The recovery
sync now includes reviewed source/tests/configs, reports, asset provenance
manifests and explicitly selected sanitized audit evidence. Runtime data, original
archives/JPEGs, checkpoints, virtual environments, cache, credentials and raw
download logs stay local. Signed URLs, tokens and secrets must never be staged.
Large local evidence trees are ignored by `recovery/.gitignore`; only individually
reviewed small proof files may be force-added. No force push is permitted.

## Future handoffs

1. Confirm authorization, destination branch, and commit author identity.
2. Verify the SSH server host key against GitHub's published fingerprints before
   trusting it; do not disable host-key verification.
3. Stage only task-related source, tests, and sanitized reports explicitly.
4. Exclude datasets, archives, checkpoints, virtual environments, credentials,
   tokens, cookies, private keys, and logs or manifests containing signed URLs.
5. Validate the changes, inspect the staged diff for secrets and large files,
   then commit and push without force.
6. Return the actual pushed commit link. Report any authorization, branch
   protection, validation, or push failure instead of claiming synchronization.

Git synchronization must not interrupt download supervisors or authorize
training. Existing recovery, full-image-audit, and five-step-smoke gates remain
unchanged.
