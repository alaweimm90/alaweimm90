"""
FEM Utilities Package
Utility functions and classes for finite element analysis.
"""
from .material_properties import (
    MaterialProperty,
    IsotropicMaterial,
    OrthotropicMaterial,
    ViscoelasticMaterial,
    MaterialLibrary,
    create_custom_material,
    scale_material_properties
)
__all__ = [
    'MaterialProperty',
    'IsotropicMaterial',
    'OrthotropicMaterial',
    'ViscoelasticMaterial',
    'MaterialLibrary',
    'create_custom_material',
    'scale_material_properties'
]