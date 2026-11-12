# Security policy

## Reporting a vulnerability

Please report security problems privately through
[GitHub's private vulnerability reporting](https://github.com/audenvenn/h3metal/security/advisories/new),
not in a public issue. You can expect an acknowledgement within a week.
Fixes are released for the latest version only.

## What is in scope

- Memory safety problems when reading a `.h3m` weight file, a safetensors
  checkpoint during conversion, or an input image.
- A downloaded file being accepted although it does not match its published
  checksum.

## What users should know

- Weight files are data, but they are mapped and parsed by native code: use
  files from sources you trust.
- Generated images are not checked for content. You are responsible for how
  you use the model and its outputs; see the model's license and acceptable
  use policy.
