# Data Contract

## Source provenance requirements

- Each source must declare:
  - name
  - uri/path
  - version or revision marker
  - weight in mixture

## Processing manifest requirements

Data processing must emit manifest entries for:

- input config path
- tokenizer artifact path
- output shard directory
- generation timestamp

## Local override policy

- Local overrides are allowed for development.
- Overrides must be explicitly declared in config and artifacts.
