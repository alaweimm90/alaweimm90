# cli
**Module:** `Python/utils/cli.py`
## Overview
Berkeley SciComp Command Line Interface
======================================
Professional command-line interface for the UC Berkeley Scientific Computing
Framework, providing unified access to all computational tools, examples,
and utilities with Berkeley branding and academic standards.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`RESET`**
- **`BOLD`**
- **`GREEN`**
- **`RED`**
- **`YELLOW`**
## Functions
### `create_parser()`
Create command-line argument parser.
**Source:** [Line 322](Python/utils/cli.py#L322)
### `main()`
Main CLI entry point.
**Source:** [Line 400](Python/utils/cli.py#L400)
## Classes
### `BerkeleyCLI`
SciComp Command Line Interface.
#### Methods
##### `__init__(self)`
*No documentation available.*
**Source:** [Line 39](Python/utils/cli.py#L39)
##### `_load_config(self)`
Load Berkeley configuration.
**Source:** [Line 44](Python/utils/cli.py#L44)
##### `_get_version(self)`
Get framework version.
**Source:** [Line 52](Python/utils/cli.py#L52)
##### `print_banner(self)`
Print Berkeley SciComp banner.
**Source:** [Line 56](Python/utils/cli.py#L56)
##### `run_quantum_physics(self, args)`
Run quantum physics simulations.
**Source:** [Line 74](Python/utils/cli.py#L74)
##### `run_ml_physics(self, args)`
Run machine learning physics simulations.
**Source:** [Line 106](Python/utils/cli.py#L106)
##### `run_quantum_computing(self, args)`
Run quantum computing algorithms.
**Source:** [Line 137](Python/utils/cli.py#L137)
##### `run_tests(self, args)`
Run framework tests.
**Source:** [Line 163](Python/utils/cli.py#L163)
##### `run_demo(self, args)`
Run framework demonstrations.
**Source:** [Line 196](Python/utils/cli.py#L196)
##### `show_config(self, args)`
Show framework configuration.
**Source:** [Line 226](Python/utils/cli.py#L226)
##### `_print_config_section(self, section, indent)`
Print configuration section with proper formatting.
**Source:** [Line 243](Python/utils/cli.py#L243)
##### `show_docs(self, args)`
Show documentation.
**Source:** [Line 259](Python/utils/cli.py#L259)
##### `apply_style(self, args)`
Apply Berkeley styling.
**Source:** [Line 301](Python/utils/cli.py#L301)
**Class Source:** [Line 36](Python/utils/cli.py#L36)