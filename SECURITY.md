# Security policy

## Credentials
- Never commit tokens, keys, or passwords to this repository.
- Never paste access tokens into chat tools, issue comments, or logs.
  Use `gh auth login` or local git credential storage for pushes.
- Any token that has appeared in plaintext anywhere must be revoked
  immediately at github.com Settings > Developer settings > Tokens.

## Data
- Production mode processes video in memory only; no frames are written
  to disk. The recorder (`cpcs_recorder.py`) is a validation-phase tool
  and its output must be treated as personal data: store locally,
  delete after labeling, never upload to third-party services.
- The SQLite database contains counts and timestamps only, no imagery
  and no personally identifying information.

## Reporting
Report vulnerabilities privately to the repository owner rather than
via public issues.
