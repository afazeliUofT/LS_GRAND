# WSL execution contract

The single wrapper verifies the package hash and repository ancestry, installs
into `/home/afazeli2006/LS_GRAND/experiments`, uses
`/home/afazeli2006/LS_GRAND/.venv`, runs tests and the gate, validates results,
stages only reviewable files, pushes the current branch, never stages `.venv`,
and never closes the VS Code terminal.
