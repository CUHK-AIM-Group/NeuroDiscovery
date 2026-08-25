# T330_tensorboard_launcher: TensorBoard Launcher for Training Runs
## Task Description

Script a TensorBoard launcher that discovers all run directories under models/, starts TensorBoard with the right logdir structure, and prints the URL.

## Input Requirement


- No interactive input.


## Constraints

- Discovers nested run dirs automatically.

- Port configurable; handles occupied ports.

- Save all generated artifacts to:
  - benchmark_results/T330_tensorboard_launcher/


## Expected Output

Expected output artifact(s):

- `launch_tensorboard.sh`

- `discovered_runs.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
