# OpenShell provider for NVIDIA NeMo Fabric

This experimental binary implements the
`fabric.environment-provider.v1alpha1` process contract. NVIDIA NeMo Fabric
starts it as `fabric-environment-openshell serve --stdio`; the process reads
newline-delimited JSON requests and returns one correlated JSON response for
each request.

The first profile supports OpenShell health, sandbox create/get/readiness,
buffered exec with bounded published output, and owned deletion. It accepts only
`in_env_control` plus `fabric_owned`, requires a digest-pinned capsule image,
and reads gateway credentials from environment-variable names supplied in
`environment.connection`. Literal token fields are rejected.

The provider accepts either a registry digest (`repository@sha256:...`) or a
local Docker image ID (`sha256:...`) for source-built development gateways.
When `settings.policy_yaml` is present, the provider parses and validates the
OpenShell policy before passing it to sandbox creation.

The provider also implements the typed `capsule_control` operation. It verifies
the bound sandbox identity, executes only
`fabric-capsule-ctl start|invoke|stop`, and rejects uncorrelated capsule
responses. The resident `fabric-capsule-runner` and configured Fabric adapter
must be installed in the digest-pinned capsule image. Environment release
remains a separate caller operation; runtime stop never deletes the sandbox.

The typed `collect_artifacts` operation accepts only adapter-declared relative
paths, executes the fixed `fabric-capsule-ctl collect-artifact` command, and
caps both individual files and the total response. It does not expose generic
remote execution to Fabric.

This first capsule profile is buffered and sequential. It deliberately omits
generic shell exposure, streaming, reconnect, and cancellation.
