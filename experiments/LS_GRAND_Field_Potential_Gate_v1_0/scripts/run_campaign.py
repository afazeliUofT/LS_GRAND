#!/usr/bin/env python3
from lsgrand.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *__import__("sys").argv[1:]]))
