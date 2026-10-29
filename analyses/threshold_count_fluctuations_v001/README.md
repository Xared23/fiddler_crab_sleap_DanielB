# Adjacent-frame count fluctuations

Methods and result links: [threshold comparison notes](../../docs/prediction_threshold_tests.md).

`calculate.py` measures signed and absolute adjacent-frame count changes for the
three completed 200-frame conditions. It validates inputs and runs seven
known-answer tests. `run.sh` executes this finite calculation through Slurm.
Job-scoped outputs contain summary statistics, every adjacent-frame increment,
and a completion/provenance receipt. Original predictions remain unchanged.
