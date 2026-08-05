# WSL execution contract

The companion file `RUN_LS_GRAND_FIELD_POTENTIAL_GATE.py` is the only wrapper
that should be placed in `/home/afazeli2006/LS_GRAND` and executed from the VS
Code WSL terminal:

```bash
cd /home/afazeli2006/LS_GRAND
python3 RUN_LS_GRAND_FIELD_POTENTIAL_GATE.py
```

The wrapper defaults to `configs/gate.json`.  It locates the exact ZIP in the
Windows Downloads directory, verifies its frozen SHA-256, safely extracts it
under `experiments/`, creates `/home/afazeli2006/LS_GRAND/.venv`, installs the
package, runs unit tests and the campaign, validates the result directory, and
commits/pushes only reviewable source and result files.

Optional environment controls:

```bash
LSGRAND_PROFILE=smoke python3 RUN_LS_GRAND_FIELD_POTENTIAL_GATE.py
LSGRAND_PROFILE=stress python3 RUN_LS_GRAND_FIELD_POTENTIAL_GATE.py
LSGRAND_SKIP_PUSH=1 python3 RUN_LS_GRAND_FIELD_POTENTIAL_GATE.py
LSGRAND_DOWNLOADS=/mnt/c/Users/alifa/Downloads python3 RUN_LS_GRAND_FIELD_POTENTIAL_GATE.py
```

The wrapper never stages `.venv`, Python caches, package build products, or
unrelated repository files.  It refuses to create a commit if unrelated files
were already staged.  On any failure it writes a timestamped log under
`results/_wrapper_logs/`, prints the exception and recovery location, and
returns control to the existing VS Code terminal; it contains no terminal-close
or shell-exit command.
