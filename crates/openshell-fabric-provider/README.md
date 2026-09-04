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

This provider does not yet run a Fabric adapter. The capsule-control transport
that binds adapter `start`, `invoke`, and `stop` is the next integration layer.
