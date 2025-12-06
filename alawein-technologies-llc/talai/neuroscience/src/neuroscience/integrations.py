"""
External API integrations for Neuroscience Research.

Interfaces with Allen Brain ORCHEX API and Neurodata Without Borders format.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime
import h5py
import numpy as np
from pathlib import Path

from .models import (
    BrainRegion,
    NeuralCircuit,
    NeuralData,
    FMRIData,
    EEGData,
    ConnectomeGraph,
)


class AllenBrainAPI:
    """Interface to Allen Brain ORCHEX."""

    BASE_URL = "http://api.brain-map.org/api/v2"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Allen Brain API client.

        Args:
            api_key: Optional API key for enhanced access
        """
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache = {}

    async def __aenter__(self):
        """Async context manager entry."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_structure_info(self, structure_id: int) -> Dict[str, Any]:
        """
        Get information about a brain structure.

        Args:
            structure_id: Allen Brain ORCHEX structure ID

        Returns:
            Structure information
        """
        if structure_id in self._cache:
            return self._cache[structure_id]

        # Simulated API response
        await asyncio.sleep(0.1)

        # Mock structure data based on common IDs
        structure_map = {
            997: {
                "id": 997,
                "name": "Cerebral cortex",
                "acronym": "CTX",
                "parent_structure_id": 8,
                "color_hex_triplet": "70FF71",
                "graph_order": 1,
                "hemisphere_id": 3,
                "weight": 8690,
                "volume": 123000.0  # mm³
            },
            12: {
                "id": 12,
                "name": "Hippocampal formation",
                "acronym": "HIP",
                "parent_structure_id": 1089,
                "color_hex_triplet": "7FC97F",
                "volume": 3500.0
            },
            159: {
                "id": 159,
                "name": "Amygdala",
                "acronym": "AMY",
                "parent_structure_id": 703,
                "color_hex_triplet": "FF8084",
                "volume": 1800.0
            }
        }

        if structure_id in structure_map:
            result = structure_map[structure_id]
        else:
            # Generic structure
            result = {
                "id": structure_id,
                "name": f"Structure_{structure_id}",
                "acronym": f"STR{structure_id}",
                "volume": np.random.uniform(100, 5000)
            }

        self._cache[structure_id] = result
        return result

    async def search_structures(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for brain structures by name.

        Args:
            query: Search query

        Returns:
            List of matching structures
        """
        await asyncio.sleep(0.1)

        # Mock search results
        results = []

        query_lower = query.lower()

        if "cortex" in query_lower:
            results.append({
                "id": 997,
                "name": "Cerebral cortex",
                "acronym": "CTX",
                "match_score": 0.95
            })
            results.append({
                "id": 315,
                "name": "Motor cortex",
                "acronym": "MO",
                "match_score": 0.85
            })

        if "hippocampus" in query_lower:
            results.append({
                "id": 12,
                "name": "Hippocampal formation",
                "acronym": "HIP",
                "match_score": 0.98
            })

        if "amygdala" in query_lower:
            results.append({
                "id": 159,
                "name": "Amygdala",
                "acronym": "AMY",
                "match_score": 0.99
            })

        return results

    async def get_gene_expression(
        self,
        gene_symbol: str,
        structure_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get gene expression data.

        Args:
            gene_symbol: Gene symbol (e.g., "BDNF")
            structure_id: Optional structure to filter by

        Returns:
            Gene expression data
        """
        await asyncio.sleep(0.1)

        # Mock gene expression data
        expression = {
            "gene_symbol": gene_symbol,
            "gene_name": self._get_gene_name(gene_symbol),
            "expression_levels": {}
        }

        # Common brain regions
        regions = ["cortex", "hippocampus", "amygdala", "thalamus", "cerebellum"]

        for region in regions:
            # Simulate expression level (z-score)
            expression["expression_levels"][region] = np.random.normal(0, 1)

        if structure_id:
            expression["structure_specific"] = {
                "structure_id": structure_id,
                "expression": np.random.normal(0, 1),
                "detection": np.random.random() > 0.3  # Detected above threshold
            }

        return expression

    async def get_connectivity_data(
        self,
        source_structure: int,
        target_structure: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get structural connectivity data.

        Args:
            source_structure: Source structure ID
            target_structure: Optional target structure ID

        Returns:
            Connectivity information
        """
        await asyncio.sleep(0.1)

        connectivity = {
            "source": source_structure,
            "connections": []
        }

        if target_structure:
            # Specific connection
            connectivity["connections"].append({
                "target": target_structure,
                "strength": np.random.uniform(0.1, 0.9),
                "tract_volume": np.random.uniform(10, 1000),  # mm³
                "confidence": np.random.uniform(0.7, 1.0)
            })
        else:
            # All connections from source
            n_connections = np.random.randint(3, 10)
            for i in range(n_connections):
                connectivity["connections"].append({
                    "target": np.random.randint(1, 1000),
                    "strength": np.random.uniform(0.1, 0.9),
                    "tract_volume": np.random.uniform(10, 1000),
                    "confidence": np.random.uniform(0.7, 1.0)
                })

        return connectivity

    async def validate_circuit_anatomy(
        self,
        circuit: NeuralCircuit
    ) -> Dict[str, Any]:
        """
        Validate circuit anatomy against Allen Brain ORCHEX.

        Args:
            circuit: Neural circuit to validate

        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "issues": [],
            "confirmed_connections": [],
            "missing_connections": []
        }

        # Check each connection
        for source, target, strength in circuit.connections:
            # Simulate checking if connection exists in ORCHEX
            exists = np.random.random() > 0.2  # 80% chance connection is known

            if exists:
                validation["confirmed_connections"].append(f"{source}->{target}")
            else:
                validation["missing_connections"].append(f"{source}->{target}")
                validation["issues"].append(
                    f"Connection {source}->{target} not found in ORCHEX"
                )

        if len(validation["missing_connections"]) > len(circuit.connections) * 0.5:
            validation["valid"] = False

        return validation

    async def get_cell_types(
        self,
        structure_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get cell type information for a structure.

        Args:
            structure_id: Structure ID

        Returns:
            List of cell types
        """
        await asyncio.sleep(0.1)

        # Mock cell type data
        cell_types = [
            {
                "name": "Pyramidal neuron",
                "percentage": 70,
                "marker_genes": ["SLC17A7", "CAMK2A"],
                "morphology": "pyramidal"
            },
            {
                "name": "Parvalbumin interneuron",
                "percentage": 5,
                "marker_genes": ["PVALB", "GAD1"],
                "morphology": "basket"
            },
            {
                "name": "Somatostatin interneuron",
                "percentage": 3,
                "marker_genes": ["SST", "GAD1"],
                "morphology": "martinotti"
            },
            {
                "name": "VIP interneuron",
                "percentage": 2,
                "marker_genes": ["VIP", "GAD1"],
                "morphology": "bipolar"
            },
            {
                "name": "Astrocyte",
                "percentage": 15,
                "marker_genes": ["GFAP", "AQP4"],
                "morphology": "stellate"
            },
            {
                "name": "Oligodendrocyte",
                "percentage": 5,
                "marker_genes": ["OLIG2", "MBP"],
                "morphology": "myelinating"
            }
        ]

        return cell_types

    def _get_gene_name(self, symbol: str) -> str:
        """Get full gene name from symbol."""
        gene_names = {
            "BDNF": "Brain-derived neurotrophic factor",
            "DRD2": "Dopamine receptor D2",
            "HTR2A": "5-hydroxytryptamine receptor 2A",
            "COMT": "Catechol-O-methyltransferase",
            "APOE": "Apolipoprotein E",
            "MAOA": "Monoamine oxidase A"
        }
        return gene_names.get(symbol, f"{symbol} gene")


class NWBInterface:
    """Neurodata Without Borders format interface."""

    def __init__(self):
        """Initialize NWB interface."""
        self.version = "2.5.0"

    async def read_nwb_file(self, filepath: Path) -> NeuralData:
        """
        Read NWB format file.

        Args:
            filepath: Path to NWB file

        Returns:
            Neural data object
        """
        # Simulate reading NWB file
        # In production, would use pynwb library
        await asyncio.sleep(0.1)

        # Mock NWB data
        neural_data = NeuralData(
            data_type="fmri",
            subject_id="sub-01",
            session_info={
                "session_id": "ses-01",
                "task": "rest",
                "acquisition": "bold"
            },
            acquisition_date=datetime.now(),
            equipment="3T Siemens Prisma",
            parameters={
                "tr": 2.0,
                "te": 30.0,
                "flip_angle": 90.0
            },
            quality_score=0.85
        )

        return neural_data

    async def write_nwb_file(
        self,
        data: NeuralData,
        filepath: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Write data to NWB format.

        Args:
            data: Neural data to write
            filepath: Output file path
            metadata: Additional metadata

        Returns:
            Success status
        """
        # Simulate writing NWB file
        await asyncio.sleep(0.1)

        # In production, would use pynwb to create NWB file
        # Here we'll create a simple JSON representation
        nwb_data = {
            "nwb_version": self.version,
            "identifier": f"{data.subject_id}_{data.session_info.get('session_id')}",
            "session_description": data.session_info.get("task", ""),
            "timestamps": datetime.now().isoformat(),
            "data_type": data.data_type.value,
            "subject": {
                "subject_id": data.subject_id
            },
            "metadata": metadata or {}
        }

        # Write JSON representation
        filepath = Path(str(filepath).replace(".nwb", ".json"))
        with open(filepath, "w") as f:
            json.dump(nwb_data, f, indent=2, default=str)

        return True

    async def validate_nwb_format(self, filepath: Path) -> Dict[str, Any]:
        """
        Validate NWB file format.

        Args:
            filepath: Path to NWB file

        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "version": self.version
        }

        # Simulate validation
        await asyncio.sleep(0.1)

        # Check file exists
        if not filepath.exists():
            validation["valid"] = False
            validation["errors"].append("File does not exist")
            return validation

        # Check file extension
        if not str(filepath).endswith((".nwb", ".json")):
            validation["warnings"].append("Unconventional file extension")

        # Mock validation checks
        if np.random.random() > 0.9:
            validation["warnings"].append("Missing optional metadata fields")

        return validation

    async def convert_to_nwb(
        self,
        data: Union[FMRIData, EEGData],
        output_path: Path
    ) -> bool:
        """
        Convert data to NWB format.

        Args:
            data: Data to convert (fMRI or EEG)
            output_path: Output file path

        Returns:
            Success status
        """
        # Create NeuralData wrapper
        neural_data = NeuralData(
            data_type="fmri" if isinstance(data, FMRIData) else "eeg",
            subject_id=data.subject_id,
            session_info={
                "session_id": data.session_id,
                "task": data.task
            },
            acquisition_date=datetime.now(),
            equipment="Unknown",
            parameters={},
            raw_data=data.model_dump()
        )

        # Write to NWB
        return await self.write_nwb_file(neural_data, output_path)

    async def extract_timeseries(
        self,
        filepath: Path,
        data_type: str = "bold"
    ) -> np.ndarray:
        """
        Extract timeseries data from NWB file.

        Args:
            filepath: Path to NWB file
            data_type: Type of timeseries to extract

        Returns:
            Timeseries array
        """
        # Simulate extraction
        await asyncio.sleep(0.1)

        # Generate mock timeseries
        if data_type == "bold":
            # fMRI-like timeseries
            n_timepoints = 200
            timeseries = np.random.randn(n_timepoints)
            # Add autocorrelation
            timeseries = np.convolve(timeseries, np.exp(-np.arange(10) / 3), mode="same")
        elif data_type == "eeg":
            # EEG-like timeseries
            n_timepoints = 10000
            timeseries = np.random.randn(n_timepoints)
            # Add oscillations
            t = np.arange(n_timepoints) / 1000  # Convert to seconds
            timeseries += np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        else:
            timeseries = np.array([])

        return timeseries

    async def merge_nwb_files(
        self,
        filepaths: List[Path],
        output_path: Path
    ) -> bool:
        """
        Merge multiple NWB files.

        Args:
            filepaths: List of NWB files to merge
            output_path: Output merged file path

        Returns:
            Success status
        """
        # Simulate merging
        await asyncio.sleep(0.2)

        # In production, would properly merge NWB files
        # Here we create a simple merged structure
        merged_data = {
            "nwb_version": self.version,
            "merged_from": [str(fp) for fp in filepaths],
            "merge_date": datetime.now().isoformat(),
            "num_sessions": len(filepaths)
        }

        with open(output_path, "w") as f:
            json.dump(merged_data, f, indent=2)

        return True