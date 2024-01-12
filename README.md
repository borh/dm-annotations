# dm-annotations

## Overview

`dm-annotations`

## Features

## Installation

`dm-annotations` uses Poetry for dependency management. To install the library, first [install Poetry](https://python-poetry.org/docs/#installation) if you haven't already. Then run:

```bash
poetry install dm-annotations
```

## Usage

### Extract DMs from corpus

```bash
python src/dm_annotations/corpus.py
```

### Convert annotations to JSONL

The converted JSONL can be used with Prodigy for further refinements.

```bash
python src/dm_annotations/corpus.py
```

### Perform network analysis and extract DM chains

```bash
python src/dm_annotations/network.py
```


### Model selection

Specifying a specific spaCy model can be done by setting the `SPACY_MODEL` environment variable.

```bash
env SPACY_MODEL=ja_ginza_bert_base python src/dm_annotations/corpus.py
```

## Development

To contribute to dm-annotations, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-repo/dm-annotations.git
cd dm-annotations
poetry install
```

Additionally, `dm-annotations` uses [devenv](https://devenv.sh/) to specify all native dependencies using Nix.
If installed, entering the directory will automatically load the right libraries and set up the Poetry virtual environment.

Note that currently, as [prodigy](https://prodi.gy/) is a hard dependency, you will not be able to use this without commenting out the dependency in `pyproject.toml`.

## Testing

`dm-annotations` uses pytest for testing. To run the tests, simply execute:

```bash
poetry run pytest
```

Additionally, many of the Python files under src contain a more extensive end-to-end test in their main.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/).
