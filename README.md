# A Novel Dynamic Gradient Calibration Method for Continual Learning

## Setup
+ git clone `git@github.com:aimagelab/mammoth.git`
+ add `DGC_ER_CIL_example.py` and `DGC_ER_TFCL_example.py` to the `models/` folder.
+ add `DGC.py` to the `models/utils/` folder.
+ add args to `utils/best_args.py` (you can copy the parameters of `er` directly and add 'DGC' : 1).
+ Use `./utils/main.py` to run experiments.
