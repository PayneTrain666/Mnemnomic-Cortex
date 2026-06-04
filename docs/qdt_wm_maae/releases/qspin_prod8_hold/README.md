# QSPIN-PROD-8-HOLD-QD6A Artifacts

This folder preserves the final QSPIN-PROD-8-HOLD pre-activation artifacts on the `experimental` branch.

## Status

- Stage: `QSPIN-PROD-8-HOLD-QD6A`
- State: final preserved pre-activation hold
- Production-active: no
- Live routing: blocked
- Real payload transfer: blocked
- Real shared-slot/QH/external-memory writes: blocked
- Commit execution: blocked
- Production activation: blocked

## Files

The GitHub connector used for this upload supports UTF-8 file creation. To preserve the binary ZIP safely, the release pack is stored as base64 chunks.

### Release pack ZIP

- `qspin_prod8_hold_qd6a_release_pack.zip.b64.part001`
- `qspin_prod8_hold_qd6a_release_pack.zip.b64.part002`
- `qspin_prod8_hold_qd6a_release_pack.zip.b64.part003`
- `qspin_prod8_hold_qd6a_release_pack.zip.b64.part004`

Reconstruct locally from this folder:

```bash
cat qspin_prod8_hold_qd6a_release_pack.zip.b64.part* | base64 -d > qspin_prod8_hold_qd6a_release_pack.zip
sha256sum qspin_prod8_hold_qd6a_release_pack.zip
```

Expected SHA-256:

```text
52921e03e26111bc61e679b796afbb9e46f47f00e76a9eb58c5346106591dacf
```

### Full printout

- `qspin_prod8_hold_full_printout.txt.part001`
- `qspin_prod8_hold_full_printout.txt.part002`
- `qspin_prod8_hold_full_printout.txt.part003`
- `qspin_prod8_hold_full_printout.txt.part004`
- `qspin_prod8_hold_full_printout.txt.part005`

Reconstruct locally from this folder:

```bash
cat qspin_prod8_hold_full_printout.txt.part* > qspin_prod8_hold_full_printout.txt
sha256sum qspin_prod8_hold_full_printout.txt
```

Expected SHA-256:

```text
6cb10091e3942d63ab3bd68ac204b74f0976a334d6ab932e00e53bc09effda7f
```

## Source workspace files

Original workspace artifact paths:

```text
/mnt/data/qspin_prod8_hold_qd6a_release_pack.zip
/mnt/data/qspin_prod8_hold_full_printout.txt
```
