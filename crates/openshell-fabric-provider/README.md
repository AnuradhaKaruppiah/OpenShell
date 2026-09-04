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

The provider also implements the typed `capsule_control` operation. It verifies
the bound sandbox identity, executes only
`fabric-capsule-ctl start|invoke|stop`, and rejects uncorrelated capsule
responses. The resident `fabric-capsule-runner` and configured Fabric adapter
must be installed in the digest-pinned capsule image. Environment release
remains a separate caller operation; runtime stop never deletes the sandbox.

This first capsule profile is buffered and sequential. It deliberately omits
generic shell exposure, streaming, reconnect, cancellation, and artifact
transfer.
