# Dry-Run Backend Interface

`BackendInterfaceProtocol` defines the dry-run write interface. `DryRunBackendInterface` stores idempotency keys in process memory only and returns a JSON-safe dry-run result describing what would have been written.

Real store writes remain blocked.
