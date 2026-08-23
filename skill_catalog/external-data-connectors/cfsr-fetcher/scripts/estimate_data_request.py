#!/usr/bin/env python3
from cfsr_fetcher import main

if __name__ == "__main__":
    raise SystemExit(main(["estimate", *__import__("sys").argv[1:]]))
