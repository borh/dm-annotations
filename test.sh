#!/usr/bin/env bash
fswatch --event Updated -o ./tests/*.py ./src/**/*.py | xargs -I{} pytest -n auto -vv
