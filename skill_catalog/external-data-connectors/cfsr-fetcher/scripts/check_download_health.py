#!/usr/bin/env python3
from cfsr_fetcher import main

if __name__ == "__main__":
    raise SystemExit(main(["health", *__import__("sys").argv[1:]]))
