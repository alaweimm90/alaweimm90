#!/usr/bin/env python3
"""
checkpoint.py - Unified Checkpoint Management Library

Provides checkpoint functionality for both governance drift detection and
orchestration workflow recovery. Consolidates logic from:
- tools/governance/checkpoint.py
- tools/orchestration/orchestration_checkpoint.py

Usage:
    from tools.lib.checkpoint import CheckpointManager
    
    # For governance drift detection
    mgr = CheckpointManager(workflow="governance")
    checkpoint_id = mgr.create_checkpoint("drift-check", context)
    
    # For orchestration recovery
    mgr = CheckpointManager(workflow="orchestration")
    checkpoint_id = mgr.create_checkpoint("feature-x", context)
"""

import json
import os
import hashlib
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict

import yaml


class CheckpointManager:
    """Unified checkpoint manager for governance and orchestration."""
    
    def __init__(self, workflow: str = "governance", base_path: Optional[Path] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            workflow: Type of workflow ("governance" or "orchestration")
            base_path: Base path for checkpoint storage
        """
        self.workflow = workflow
        self.base_path = base_path or self._find_base_path()
        
        # Set checkpoint directory based on workflow type
        if workflow == "governance":
            self.checkpoint_dir = self.base_path / ".metaHub/checkpoints"
            self.snapshot_dir = None
        else:  # orchestration
            self.checkpoint_dir = self.base_path / ".metaHub/orchestration/checkpoints"
            self.snapshot_dir = self.base_path / ".metaHub/orchestration/snapshots"
            
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def _find_base_path(self) -> Path:
        """Find the central governance repo path."""
        if env_path := os.environ.get("GOLDEN_PATH_ROOT"):
            path = Path(env_path)
            if path.exists() and (path / ".metaHub").exists():
                return path
        
        current = Path.cwd()
        while current != current.parent:
            if (current / ".metaHub").exists():
                return current
            current = current.parent
        
        script_path = Path(__file__).resolve().parent.parent.parent
        if (script_path / ".metaHub").exists():
            return script_path
        
        raise RuntimeError("Could not find central governance repo")
    
    def create_checkpoint(self, workflow: str, context: dict) -> str:
        """
        Create a new checkpoint.
        
        Args:
            workflow: Workflow name/identifier
            context: Context data to checkpoint
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "workflow": workflow,
            "created_at": timestamp,
            "context": context,
            "checksum": ""
        }
        
        # Generate checksum
        data_copy = {k: v for k, v in checkpoint_data.items() if k != "checksum"}
        json_str = json.dumps(data_copy, sort_keys=True, default=str)
        checkpoint_data["checksum"] = hashlib.sha256(json_str.encode()).hexdigest()[:16]
        
        # Save checkpoint
        filename = f"{workflow}_{checkpoint_id}.json"
        filepath = self.checkpoint_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        # Update latest pointer
        latest_file = self.checkpoint_dir / f"{workflow}_latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> dict:
        """
        Restore a checkpoint by ID.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            
        Returns:
            Checkpoint data dictionary
        """
        # Search for checkpoint file
        for filepath in self.checkpoint_dir.glob(f"*_{checkpoint_id}.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        
        raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
    
    def list_checkpoints(self, workflow: str = None) -> list:
        """
        List available checkpoints.
        
        Args:
            workflow: Optional workflow filter
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        
        pattern = f"{workflow}_*.json" if workflow else "*.json"
        for filepath in self.checkpoint_dir.glob(pattern):
            if filepath.name.endswith("_latest.json"):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                checkpoints.append({
                    "id": data.get("checkpoint_id"),
                    "workflow": data.get("workflow"),
                    "created_at": data.get("created_at"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Sort by creation time descending
        checkpoints.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return checkpoints
    
    def validate_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint_id: ID of checkpoint to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            data = self.restore_checkpoint(checkpoint_id)
            
            # Verify checksum
            stored_checksum = data.get("checksum", "")
            data_copy = {k: v for k, v in data.items() if k != "checksum"}
            json_str = json.dumps(data_copy, sort_keys=True, default=str)
            calculated_checksum = hashlib.sha256(json_str.encode()).hexdigest()[:16]
            
            return stored_checksum == calculated_checksum
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
    
    def cleanup_old(self, retention_days: int = 30) -> int:
        """
        Remove checkpoints older than retention period.
        
        Args:
            retention_days: Number of days to retain
            
        Returns:
            Number of checkpoints removed
        """
        cutoff = datetime.now() - timedelta(days=retention_days)
        removed = 0
        
        for filepath in self.checkpoint_dir.glob("*.json"):
            if filepath.name.endswith("_latest.json"):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                created_at = datetime.fromisoformat(data["created_at"])
                if created_at < cutoff:
                    filepath.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        
        return removed