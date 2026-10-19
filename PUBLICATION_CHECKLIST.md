# GitHub Publication Checklist

The working tree is prepared for a source-focused GitHub repository, but the
items below need an owner decision before public release.

- **Repository license:** there is no root `LICENSE` file. The original MNSIM
  source attribution and its redistribution terms need confirmation before a
  repository-wide license is added. Do not assume a third-party model's license
  also licenses MNSIM code.
- **Third-party source review:** inspect every bundled source/model and its
  license. The Stanford-PKU source is MemTorch-derived and GPL-3.0. Preserve
  its notice and determine whether redistribution is compatible with the
  intended license.
- **ASU model redistribution:** confirm that the Stanford/ASU model package and
  generated artifacts may be redistributed, and preserve the upstream
  attribution and applicable terms.
- **Paper and dataset rights:** papers, publisher templates, manuscript drafts,
  datasets, and trained checkpoints are not part of the curated GitHub source
  release by default. Review individual rights before adding any of them.
- **Git remote and release identity:** no Git remote is configured. Confirm the
  destination organization, repository name, visibility, and release tag before
  publishing.
- **Final content review:** inspect the staged file list and scan the complete
  Git history for secrets and material that should not be public. `.gitignore`
  protects local/generated files from new additions; it does not remove files
  already committed in history.

Experiment summaries, per-step ledgers, schedules, deployment weights, training
artifacts, logs, and local environment snapshots are excluded from the source
release.
