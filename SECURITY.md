# Security Policy

## Reporting a Vulnerability

If you believe you have found a security vulnerability in mlir-aie, please
report it privately rather than opening a public issue.

The preferred channel is GitHub's private vulnerability reporting:

  https://github.com/Xilinx/mlir-aie/security/advisories/new

This opens a private advisory thread visible only to the maintainers. You
should expect an initial acknowledgement within 5 business days.

Please include, where possible:
- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a minimal proof-of-concept.
- The commit hash or release tag the issue was observed against.
- Any suggested mitigation.

## Supported Versions

mlir-aie tracks the tip of `main`. Security fixes are applied to `main` and
will appear in the next wheel build; older wheel releases are not patched
in place.

## Scope

In-scope: code in this repository, including build tooling, Python
bindings, and CI workflows.

Out-of-scope: vulnerabilities in upstream LLVM/MLIR (please report to the
LLVM project), in the AMD XDNA driver (please report through AMD's
disclosure channel), or in third-party dependencies (please report to the
respective project).
