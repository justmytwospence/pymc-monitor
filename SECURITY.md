# Security policy

For private reports, open a GitHub security advisory on the repository rather
than filing a public issue. The local dashboard binds only to `127.0.0.1`,
requires a unique bearer token for state reads and stop requests, and ships its
scripts locally. Treat dashboard URLs as secrets; a local process with access
to the URL can view run progress and request a stop. Close the handle to stop
the service. No credentials or posterior samples are sent to a remote service.
