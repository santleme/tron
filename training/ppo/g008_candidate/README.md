# Frozen G008 benchmark candidate

`train.py` is preserved byte-for-byte from commit
`4849bb95d255af186225c1006f29bc66fb832427` (SHA-256
`9991dc8e0c45cf586d2597f82f1cc30ceaf62c6754b976bd39eca70a3f86c38c`). The
paired benchmark loads this snapshot so later trainer changes cannot silently
change the candidate being measured. Its engine dependency is the separately
verified `training/arena/engine.py` snapshot with SHA-256
`cfd6fe1817e06e5e892306f1fac2b5eef3cdeb4e68da2e83b26c8f4f23a6c7d0`.

Do not edit these frozen bytes. A different candidate needs a new source
identity and benchmark registration.
