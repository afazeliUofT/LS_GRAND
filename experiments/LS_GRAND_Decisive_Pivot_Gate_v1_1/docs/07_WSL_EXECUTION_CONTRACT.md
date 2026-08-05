# WSL execution contract

- Repository: `/home/afazeli2006/LS_GRAND`
- Expected origin: `afazeliUofT/LS_GRAND`
- Required ancestor: `531fdae2c27cb798edfe1bff6a8269f2fa341e29`
- Local environment: `/home/afazeli2006/LS_GRAND/.venv`
- Package installation: `experiments/LS_GRAND_Decisive_Pivot_Gate_v1_1`
- Results: `results/LS_GRAND_Decisive_Pivot_Gate_<profile>_<UTC>`

The wrapper refuses to mix pre-existing staged files into its commit.  It stages
only the wrapper, package source/docs/configuration, validation artifacts, and
the new result directory.  `.venv`, caches, editable-build products, and
unrelated repository files are excluded.  No command closes or replaces the VS
Code terminal.
