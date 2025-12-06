"""
File parsers for MagLogic package.

This module provides parsers for various micromagnetic simulation output formats
including OOMMF OVF/ODT files and MuMax3 output files.

Author: Meshal Alawein
Email: meshal@berkeley.edu
"""

from .oommf_parser import OOMMFParser
from .mumax3_parser import MuMax3Parser
from .base_parser import BaseParser

__all__ = ["OOMMFParser", "MuMax3Parser", "BaseParser"]