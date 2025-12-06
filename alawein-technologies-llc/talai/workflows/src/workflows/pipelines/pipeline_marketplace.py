"""
Pipeline Marketplace - Save, share, and discover validation pipelines

Marketplace for sharing and reusing validation pipelines across projects.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

from .validation_pipeline import ValidationPipeline


logger = logging.getLogger(__name__)


@dataclass
class PipelineMetadata:
    """Metadata for marketplace pipeline"""
    id: str
    name: str
    description: str
    author: str
    version: str
    category: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    downloads: int = 0
    rating: float = 0.0
    reviews: List[Dict[str, Any]] = field(default_factory=list)


class PipelineMarketplace:
    """Marketplace for validation pipelines"""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path.cwd() / ".pipelines"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.pipelines: Dict[str, PipelineMetadata] = {}
        self.pipeline_instances: Dict[str, ValidationPipeline] = {}
        self._load_marketplace()

    def publish_pipeline(self, pipeline: ValidationPipeline,
                        metadata: PipelineMetadata) -> str:
        """Publish pipeline to marketplace"""
        # Save pipeline configuration
        pipeline_config = pipeline.to_dict()

        # Store metadata
        self.pipelines[metadata.id] = metadata
        self.pipeline_instances[metadata.id] = pipeline

        # Save to disk
        self._save_pipeline(metadata.id, pipeline_config, metadata)

        logger.info(f"Published pipeline: {metadata.name} (ID: {metadata.id})")
        return metadata.id

    def get_pipeline(self, pipeline_id: str) -> Optional[ValidationPipeline]:
        """Get pipeline from marketplace"""
        if pipeline_id in self.pipeline_instances:
            # Update download count
            self.pipelines[pipeline_id].downloads += 1
            self._update_metadata(pipeline_id)
            return self.pipeline_instances[pipeline_id]

        # Try to load from disk
        pipeline = self._load_pipeline(pipeline_id)
        if pipeline:
            self.pipeline_instances[pipeline_id] = pipeline
            return pipeline

        return None

    def search_pipelines(self, query: Optional[str] = None,
                        category: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> List[PipelineMetadata]:
        """Search for pipelines in marketplace"""
        results = list(self.pipelines.values())

        # Filter by query
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.name.lower() or query_lower in p.description.lower()
            ]

        # Filter by category
        if category:
            results = [p for p in results if p.category == category]

        # Filter by tags
        if tags:
            results = [
                p for p in results
                if any(tag in p.tags for tag in tags)
            ]

        # Sort by rating and downloads
        results.sort(key=lambda p: (p.rating, p.downloads), reverse=True)

        return results

    def rate_pipeline(self, pipeline_id: str, rating: float,
                      review: Optional[str] = None, reviewer: str = "anonymous"):
        """Rate a pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        metadata = self.pipelines[pipeline_id]

        # Add review
        if review:
            metadata.reviews.append({
                "rating": rating,
                "review": review,
                "reviewer": reviewer,
                "timestamp": datetime.now().isoformat()
            })

        # Update average rating
        all_ratings = [r["rating"] for r in metadata.reviews if "rating" in r]
        all_ratings.append(rating)
        metadata.rating = sum(all_ratings) / len(all_ratings)

        # Save updated metadata
        self._update_metadata(pipeline_id)

        logger.info(f"Rated pipeline {pipeline_id}: {rating} stars")

    def fork_pipeline(self, pipeline_id: str, new_name: str,
                     author: str) -> str:
        """Fork an existing pipeline"""
        original = self.get_pipeline(pipeline_id)
        if not original:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        # Create new metadata
        new_id = f"{pipeline_id}_fork_{datetime.now().timestamp()}"
        metadata = PipelineMetadata(
            id=new_id,
            name=new_name,
            description=f"Fork of {self.pipelines[pipeline_id].name}",
            author=author,
            version="1.0.0",
            category=self.pipelines[pipeline_id].category,
            tags=self.pipelines[pipeline_id].tags + ["fork"]
        )

        # Create copy of pipeline
        forked_pipeline = ValidationPipeline(
            name=new_name,
            description=metadata.description
        )
        forked_pipeline.steps = original.steps.copy()
        forked_pipeline.pre_processors = original.pre_processors.copy()
        forked_pipeline.post_processors = original.post_processors.copy()

        # Publish forked pipeline
        return self.publish_pipeline(forked_pipeline, metadata)

    def get_trending_pipelines(self, limit: int = 10) -> List[PipelineMetadata]:
        """Get trending pipelines"""
        # Sort by recent downloads and rating
        sorted_pipelines = sorted(
            self.pipelines.values(),
            key=lambda p: (p.downloads * 0.3 + p.rating * 0.7),
            reverse=True
        )

        return sorted_pipelines[:limit]

    def get_categories(self) -> List[str]:
        """Get all pipeline categories"""
        categories = set(p.category for p in self.pipelines.values())
        return sorted(list(categories))

    def get_popular_tags(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get popular tags with counts"""
        tag_counts = {}

        for pipeline in self.pipelines.values():
            for tag in pipeline.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Sort by count
        sorted_tags = sorted(
            tag_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {"tag": tag, "count": count}
            for tag, count in sorted_tags[:limit]
        ]

    def _save_pipeline(self, pipeline_id: str, config: Dict[str, Any],
                      metadata: PipelineMetadata):
        """Save pipeline to disk"""
        filepath = self.storage_path / f"{pipeline_id}.json"

        data = {
            "config": config,
            "metadata": {
                "id": metadata.id,
                "name": metadata.name,
                "description": metadata.description,
                "author": metadata.author,
                "version": metadata.version,
                "category": metadata.category,
                "tags": metadata.tags,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat(),
                "downloads": metadata.downloads,
                "rating": metadata.rating,
                "reviews": metadata.reviews
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_pipeline(self, pipeline_id: str) -> Optional[ValidationPipeline]:
        """Load pipeline from disk"""
        filepath = self.storage_path / f"{pipeline_id}.json"

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Reconstruct pipeline
            config = data["config"]
            pipeline = ValidationPipeline(
                name=config["name"],
                description=config["description"]
            )

            # Note: Steps need to be reconstructed with actual validators
            # This is a simplified version
            pipeline.metadata = config.get("metadata", {})

            return pipeline

        except Exception as e:
            logger.error(f"Failed to load pipeline {pipeline_id}: {e}")
            return None

    def _load_marketplace(self):
        """Load all pipelines from storage"""
        if not self.storage_path.exists():
            return

        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                metadata_dict = data["metadata"]
                metadata = PipelineMetadata(
                    id=metadata_dict["id"],
                    name=metadata_dict["name"],
                    description=metadata_dict["description"],
                    author=metadata_dict["author"],
                    version=metadata_dict["version"],
                    category=metadata_dict["category"],
                    tags=metadata_dict["tags"],
                    created_at=datetime.fromisoformat(metadata_dict["created_at"]),
                    updated_at=datetime.fromisoformat(metadata_dict["updated_at"]),
                    downloads=metadata_dict["downloads"],
                    rating=metadata_dict["rating"],
                    reviews=metadata_dict["reviews"]
                )

                self.pipelines[metadata.id] = metadata

            except Exception as e:
                logger.error(f"Failed to load marketplace entry from {filepath}: {e}")

    def _update_metadata(self, pipeline_id: str):
        """Update metadata for pipeline"""
        if pipeline_id not in self.pipelines:
            return

        metadata = self.pipelines[pipeline_id]
        metadata.updated_at = datetime.now()

        # Load existing pipeline config
        filepath = self.storage_path / f"{pipeline_id}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)

            config = data["config"]
            self._save_pipeline(pipeline_id, config, metadata)

    def export_catalog(self) -> Dict[str, Any]:
        """Export marketplace catalog"""
        return {
            "pipelines": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "author": p.author,
                    "category": p.category,
                    "tags": p.tags,
                    "rating": p.rating,
                    "downloads": p.downloads
                }
                for p in self.pipelines.values()
            ],
            "categories": self.get_categories(),
            "popular_tags": self.get_popular_tags()
        }