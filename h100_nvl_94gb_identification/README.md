# CLAIX H100 Hardware Identification

This working directory contains a read-only hardware inventory script for the
CLAIX-2023 ML-node H100 GPUs. The version-controlled thesis copy is under
`~/Master-Thesis/gitlab/sourcecode/h100_nvl_94gb_identification/`.

Run the collector for physical GPU 3:

```bash
./collect_h100_inventory.sh 3
```

It writes a timestamped text report in the current directory unless an output
path is supplied as the second argument. CUDA attributes are addressed through
the physical GPU UUID, rather than a CUDA ordinal, so a MIG slice cannot be
mistaken for the requested physical GPU.
