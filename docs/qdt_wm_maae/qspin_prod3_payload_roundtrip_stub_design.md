# Payload Roundtrip Stub Design

Payload roundtrip stubs are metadata/synthetic only. Dense, cHRR, and QH payload metadata produce deterministic metadata-only roundtrip hashes. Raw payloads, tensors, transfer, storage, writes, unsafe shapes, budget exceedance, and invalid norm bands are rejected.
