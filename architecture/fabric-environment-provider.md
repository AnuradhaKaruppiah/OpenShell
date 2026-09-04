# NVIDIA NeMo Fabric environment provider

OpenShell supplies an experimental out-of-process environment provider for
NVIDIA NeMo Fabric. Keeping the provider process outside Fabric core prevents
OpenShell's asynchronous transport, authentication, and gRPC dependencies from
becoming mandatory Fabric dependencies.

Fabric starts `fabric-environment-openshell serve --stdio` and exchanges
newline-delimited `fabric.environment-provider.v1alpha1` messages. Fabric owns
provider selection, correlation, normalized environment handles, and the
future adapter lifecycle. The provider owns OpenShell gateway health checks,
sandbox identity, readiness, response-bounded exec, and ownership-aware
deletion.

The initial profile is deliberately narrow:

- `control_location` must be `in_env_control`;
- `ownership` must be `fabric_owned`;
- capsule images must use an immutable SHA-256 digest;
- gateway bearer tokens and private CA material are named by environment
  variable references rather than included in Fabric plans; and
- exec is buffered with a capped published response, not advertised as
  streaming.

The protocol is experimental. A later descriptor and persistent process may
stabilize it after the capsule-control path proves stateful adapter execution.
