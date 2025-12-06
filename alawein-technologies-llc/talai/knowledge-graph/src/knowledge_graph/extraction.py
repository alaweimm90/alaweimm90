"""
Entity and relation extraction for knowledge graph construction.

Implements NER, relation extraction, and triple extraction from scientific papers.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    text: str
    type: str  # 'method', 'dataset', 'metric', 'concept', 'author', 'institution'
    confidence: float
    context: str
    source_paper: str
    attributes: Dict[str, Any] = None


@dataclass
class Relation:
    """Represents a relation between entities."""
    id: str
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    evidence: str
    source_paper: str
    attributes: Dict[str, Any] = None


@dataclass
class Triple:
    """Knowledge graph triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float
    provenance: Dict[str, Any]


class EntityExtractor:
    """Extract entities from scientific text."""

    def __init__(self, entity_types: Optional[List[str]] = None):
        """Initialize entity extractor.

        Args:
            entity_types: Types of entities to extract
        """
        self.entity_types = entity_types or [
            'method', 'dataset', 'metric', 'concept',
            'author', 'institution', 'software', 'task'
        ]
        self.patterns = self._compile_patterns()
        self.entity_cache = {}

    def extract_entities(self, text: str, paper_id: str) -> List[Entity]:
        """Extract entities from text.

        Args:
            text: Input text
            paper_id: Source paper ID

        Returns:
            List of extracted entities
        """
        entities = []

        # Method extraction
        methods = self._extract_methods(text)
        for method in methods:
            entities.append(Entity(
                id=f"{paper_id}_method_{len(entities)}",
                text=method['text'],
                type='method',
                confidence=method['confidence'],
                context=method['context'],
                source_paper=paper_id,
                attributes={'algorithm_type': method.get('algorithm_type')}
            ))

        # Dataset extraction
        datasets = self._extract_datasets(text)
        for dataset in datasets:
            entities.append(Entity(
                id=f"{paper_id}_dataset_{len(entities)}",
                text=dataset['text'],
                type='dataset',
                confidence=dataset['confidence'],
                context=dataset['context'],
                source_paper=paper_id,
                attributes={'dataset_size': dataset.get('size')}
            ))

        # Metric extraction
        metrics = self._extract_metrics(text)
        for metric in metrics:
            entities.append(Entity(
                id=f"{paper_id}_metric_{len(entities)}",
                text=metric['text'],
                type='metric',
                confidence=metric['confidence'],
                context=metric['context'],
                source_paper=paper_id,
                attributes={'value': metric.get('value')}
            ))

        # Concept extraction
        concepts = self._extract_concepts(text)
        for concept in concepts:
            entities.append(Entity(
                id=f"{paper_id}_concept_{len(entities)}",
                text=concept['text'],
                type='concept',
                confidence=concept['confidence'],
                context=concept['context'],
                source_paper=paper_id
            ))

        # Author extraction
        authors = self._extract_authors(text)
        for author in authors:
            entities.append(Entity(
                id=f"{paper_id}_author_{len(entities)}",
                text=author['text'],
                type='author',
                confidence=author['confidence'],
                context=author['context'],
                source_paper=paper_id,
                attributes={'affiliation': author.get('affiliation')}
            ))

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        return entities

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for entity extraction."""
        patterns = {
            'method': [
                re.compile(r'\b(?:algorithm|method|approach|technique|model)\s+called\s+(\w+)', re.I),
                re.compile(r'\b(?:propose|introduce|present)\s+(?:a\s+)?(?:new\s+)?(\w+\s+\w+)', re.I),
                re.compile(r'\b([A-Z][A-Za-z]+(?:[A-Z][A-Za-z]+)*)\s+(?:algorithm|method)', re.I),
            ],
            'dataset': [
                re.compile(r'\b(?:dataset|corpus|benchmark)\s+(?:called\s+)?(\w+)', re.I),
                re.compile(r'\b([A-Z][A-Za-z]+(?:-[A-Z][A-Za-z]+)*)\s+dataset', re.I),
                re.compile(r'\b(?:ImageNet|COCO|MNIST|CIFAR|WikiText|GLUE|SQuAD)\b', re.I),
            ],
            'metric': [
                re.compile(r'\b(?:accuracy|precision|recall|F1|BLEU|ROUGE|perplexity)\s*(?:of\s+)?([0-9.]+)', re.I),
                re.compile(r'\b([0-9.]+)%?\s+(?:accuracy|precision|recall|F1)', re.I),
                re.compile(r'\b(?:achieve|obtain|reach)\s+([0-9.]+)%?\s+(?:on|in)', re.I),
            ],
            'author': [
                re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+et\s+al\.'),
                re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\(\d{4}\)'),
            ]
        }
        return patterns

    def _extract_methods(self, text: str) -> List[Dict[str, Any]]:
        """Extract method/algorithm entities."""
        methods = []

        for pattern in self.patterns.get('method', []):
            matches = pattern.finditer(text)
            for match in matches:
                method_text = match.group(1).strip()
                context = text[max(0, match.start()-50):min(len(text), match.end()+50)]

                methods.append({
                    'text': method_text,
                    'confidence': 0.8,
                    'context': context,
                    'algorithm_type': self._classify_algorithm(method_text)
                })

        # Look for known algorithm names
        known_algorithms = [
            'BERT', 'GPT', 'Transformer', 'LSTM', 'CNN', 'RNN', 'GAN',
            'VAE', 'ResNet', 'VGG', 'YOLO', 'R-CNN', 'SVM', 'Random Forest'
        ]

        for algo in known_algorithms:
            if re.search(r'\b' + algo + r'\b', text, re.I):
                context_match = re.search(r'.{0,50}\b' + algo + r'\b.{0,50}', text, re.I)
                if context_match:
                    methods.append({
                        'text': algo,
                        'confidence': 0.9,
                        'context': context_match.group(),
                        'algorithm_type': self._classify_algorithm(algo)
                    })

        return methods

    def _extract_datasets(self, text: str) -> List[Dict[str, Any]]:
        """Extract dataset entities."""
        datasets = []

        for pattern in self.patterns.get('dataset', []):
            matches = pattern.finditer(text)
            for match in matches:
                if len(match.groups()) > 0:
                    dataset_text = match.group(1).strip()
                else:
                    dataset_text = match.group().strip()

                context = text[max(0, match.start()-50):min(len(text), match.end()+50)]

                # Extract dataset size if mentioned
                size_match = re.search(r'(\d+(?:,\d+)*)\s+(?:samples|examples|instances)', context)
                size = size_match.group(1) if size_match else None

                datasets.append({
                    'text': dataset_text,
                    'confidence': 0.85,
                    'context': context,
                    'size': size
                })

        return datasets

    def _extract_metrics(self, text: str) -> List[Dict[str, Any]]:
        """Extract metric entities."""
        metrics = []

        for pattern in self.patterns.get('metric', []):
            matches = pattern.finditer(text)
            for match in matches:
                metric_value = match.group(1).strip()
                context = text[max(0, match.start()-50):min(len(text), match.end()+50)]

                # Determine metric type
                metric_type = 'unknown'
                for mtype in ['accuracy', 'precision', 'recall', 'F1', 'BLEU', 'ROUGE', 'perplexity']:
                    if mtype.lower() in context.lower():
                        metric_type = mtype
                        break

                metrics.append({
                    'text': f"{metric_type}={metric_value}",
                    'confidence': 0.75,
                    'context': context,
                    'value': metric_value,
                    'metric_type': metric_type
                })

        return metrics

    def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract scientific concepts using TF-IDF."""
        concepts = []

        # Simple noun phrase extraction
        noun_phrases = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\b',
            text
        )

        # Filter by length and common words
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'We', 'Our', 'In', 'For'}

        for phrase in noun_phrases:
            if len(phrase.split()) <= 3 and phrase not in stop_words:
                context_match = re.search(
                    r'.{0,50}\b' + re.escape(phrase) + r'\b.{0,50}',
                    text,
                    re.I
                )

                if context_match:
                    concepts.append({
                        'text': phrase,
                        'confidence': 0.6,
                        'context': context_match.group()
                    })

        return concepts[:20]  # Limit to top 20 concepts

    def _extract_authors(self, text: str) -> List[Dict[str, Any]]:
        """Extract author entities."""
        authors = []

        for pattern in self.patterns.get('author', []):
            matches = pattern.finditer(text)
            for match in matches:
                author_text = match.group(1).strip()
                context = text[max(0, match.start()-50):min(len(text), match.end()+50)]

                # Try to extract affiliation
                affiliation = None
                affil_match = re.search(
                    author_text + r'[^.]*(?:University|Institute|Laboratory|Lab)',
                    text,
                    re.I
                )
                if affil_match:
                    affiliation = affil_match.group()

                authors.append({
                    'text': author_text,
                    'confidence': 0.7,
                    'context': context,
                    'affiliation': affiliation
                })

        return authors

    def _classify_algorithm(self, algorithm_name: str) -> str:
        """Classify algorithm type."""
        algorithm_name = algorithm_name.lower()

        if any(term in algorithm_name for term in ['neural', 'deep', 'transformer', 'bert', 'gpt']):
            return 'deep_learning'
        elif any(term in algorithm_name for term in ['svm', 'forest', 'tree', 'bayes', 'knn']):
            return 'traditional_ml'
        elif any(term in algorithm_name for term in ['genetic', 'evolutionary', 'swarm']):
            return 'evolutionary'
        elif any(term in algorithm_name for term in ['reinforce', 'q-learning', 'policy']):
            return 'reinforcement_learning'
        else:
            return 'other'

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen = set()
        deduplicated = []

        for entity in entities:
            key = (entity.text.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)

        return deduplicated


class RelationExtractor:
    """Extract relations between entities."""

    def __init__(self):
        """Initialize relation extractor."""
        self.relation_patterns = self._compile_relation_patterns()
        self.predicate_map = self._create_predicate_map()

    def extract_relations(self, entities: List[Entity], text: str,
                         paper_id: str) -> List[Relation]:
        """Extract relations between entities.

        Args:
            entities: List of entities
            text: Source text
            paper_id: Paper ID

        Returns:
            List of extracted relations
        """
        relations = []

        # Create entity position map
        entity_positions = self._map_entity_positions(entities, text)

        # Extract relations for each entity pair
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j:  # Skip self-relations and duplicates
                    continue

                # Check if entities are close enough
                if self._entities_nearby(entity1, entity2, entity_positions, threshold=100):
                    # Extract relation
                    relation = self._extract_relation_between(
                        entity1, entity2, text, entity_positions
                    )

                    if relation:
                        relations.append(Relation(
                            id=f"{paper_id}_rel_{len(relations)}",
                            subject=entity1,
                            predicate=relation['predicate'],
                            object=entity2,
                            confidence=relation['confidence'],
                            evidence=relation['evidence'],
                            source_paper=paper_id,
                            attributes=relation.get('attributes')
                        ))

        # Extract typed relations
        typed_relations = self._extract_typed_relations(entities, text, paper_id)
        relations.extend(typed_relations)

        return relations

    def _compile_relation_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile patterns for relation extraction."""
        patterns = {
            'uses': [
                re.compile(r'(\w+)\s+(?:uses?|utilizes?|employs?)\s+(\w+)'),
                re.compile(r'(\w+)\s+(?:based on|built on)\s+(\w+)'),
            ],
            'improves': [
                re.compile(r'(\w+)\s+(?:improves?|outperforms?|beats?)\s+(\w+)'),
                re.compile(r'(\w+)\s+(?:better than|superior to)\s+(\w+)'),
            ],
            'evaluates': [
                re.compile(r'(\w+)\s+(?:evaluated on|tested on)\s+(\w+)'),
                re.compile(r'evaluate\s+(\w+)\s+on\s+(\w+)'),
            ],
            'proposes': [
                re.compile(r'(?:propose|introduce|present)\s+(\w+)'),
            ],
            'achieves': [
                re.compile(r'(\w+)\s+achieves?\s+([0-9.]+)'),
                re.compile(r'(\w+)\s+(?:obtains?|reaches?)\s+([0-9.]+)'),
            ]
        }
        return patterns

    def _create_predicate_map(self) -> Dict[str, str]:
        """Create mapping of relation types to predicates."""
        return {
            'uses': 'uses_technique',
            'improves': 'outperforms',
            'evaluates': 'evaluated_on',
            'proposes': 'introduces',
            'achieves': 'achieves_score',
            'implements': 'implements',
            'extends': 'extends',
            'compares': 'compared_with',
            'trains': 'trained_on',
            'requires': 'requires'
        }

    def _map_entity_positions(self, entities: List[Entity],
                             text: str) -> Dict[str, List[int]]:
        """Map entities to their positions in text."""
        positions = {}

        for entity in entities:
            entity_positions = []
            pattern = re.compile(r'\b' + re.escape(entity.text) + r'\b', re.I)

            for match in pattern.finditer(text):
                entity_positions.append((match.start(), match.end()))

            positions[entity.id] = entity_positions

        return positions

    def _entities_nearby(self, entity1: Entity, entity2: Entity,
                        positions: Dict[str, List[int]],
                        threshold: int = 100) -> bool:
        """Check if two entities are nearby in text."""
        pos1 = positions.get(entity1.id, [])
        pos2 = positions.get(entity2.id, [])

        if not pos1 or not pos2:
            return False

        # Check minimum distance between any occurrences
        for p1 in pos1:
            for p2 in pos2:
                distance = min(abs(p1[0] - p2[1]), abs(p2[0] - p1[1]))
                if distance <= threshold:
                    return True

        return False

    def _extract_relation_between(self, entity1: Entity, entity2: Entity,
                                 text: str, positions: Dict[str, List[int]]) -> Optional[Dict]:
        """Extract relation between two entities."""
        # Get text between entities
        pos1 = positions.get(entity1.id, [])
        pos2 = positions.get(entity2.id, [])

        if not pos1 or not pos2:
            return None

        # Find closest occurrence
        min_distance = float('inf')
        best_context = ""

        for p1 in pos1:
            for p2 in pos2:
                if p1[1] < p2[0]:  # entity1 before entity2
                    between = text[p1[1]:p2[0]]
                    distance = p2[0] - p1[1]
                elif p2[1] < p1[0]:  # entity2 before entity1
                    between = text[p2[1]:p1[0]]
                    distance = p1[0] - p2[1]
                else:
                    continue

                if distance < min_distance and distance < 100:
                    min_distance = distance
                    best_context = between

        if not best_context:
            return None

        # Determine relation type
        predicate = self._determine_predicate(best_context)

        if predicate:
            return {
                'predicate': predicate,
                'confidence': 0.7,
                'evidence': best_context.strip(),
                'attributes': {'distance': min_distance}
            }

        return None

    def _determine_predicate(self, context: str) -> Optional[str]:
        """Determine predicate from context."""
        context_lower = context.lower()

        # Check for known relation keywords
        if any(word in context_lower for word in ['use', 'utilize', 'employ']):
            return 'uses_technique'
        elif any(word in context_lower for word in ['improve', 'outperform', 'beat']):
            return 'outperforms'
        elif any(word in context_lower for word in ['evaluate', 'test', 'benchmark']):
            return 'evaluated_on'
        elif any(word in context_lower for word in ['extend', 'build on', 'based on']):
            return 'extends'
        elif any(word in context_lower for word in ['compare', 'versus', 'vs']):
            return 'compared_with'
        elif any(word in context_lower for word in ['train', 'fine-tune']):
            return 'trained_on'
        elif any(word in context_lower for word in ['require', 'need', 'depend']):
            return 'requires'

        return None

    def _extract_typed_relations(self, entities: List[Entity], text: str,
                                paper_id: str) -> List[Relation]:
        """Extract relations of specific types."""
        relations = []

        # Method-Dataset relations
        methods = [e for e in entities if e.type == 'method']
        datasets = [e for e in entities if e.type == 'dataset']

        for method in methods:
            for dataset in datasets:
                # Check if method is evaluated on dataset
                pattern = re.compile(
                    method.text + r'.*(?:on|using)\s+' + dataset.text,
                    re.I
                )
                if pattern.search(text):
                    relations.append(Relation(
                        id=f"{paper_id}_rel_{len(relations)}",
                        subject=method,
                        predicate='evaluated_on',
                        object=dataset,
                        confidence=0.8,
                        evidence=f"{method.text} evaluated on {dataset.text}",
                        source_paper=paper_id
                    ))

        # Method-Metric relations
        metrics = [e for e in entities if e.type == 'metric']

        for method in methods:
            for metric in metrics:
                # Check if method achieves metric
                if method.text in metric.context or metric.text in method.context:
                    relations.append(Relation(
                        id=f"{paper_id}_rel_{len(relations)}",
                        subject=method,
                        predicate='achieves_score',
                        object=metric,
                        confidence=0.75,
                        evidence=f"{method.text} achieves {metric.text}",
                        source_paper=paper_id
                    ))

        return relations


class TripleExtractor:
    """Extract knowledge graph triples."""

    def __init__(self):
        """Initialize triple extractor."""
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()

    def extract_triples(self, text: str, paper_id: str,
                       metadata: Optional[Dict[str, Any]] = None) -> List[Triple]:
        """Extract triples from text.

        Args:
            text: Input text
            paper_id: Paper ID
            metadata: Optional paper metadata

        Returns:
            List of extracted triples
        """
        # Extract entities
        entities = self.entity_extractor.extract_entities(text, paper_id)

        # Extract relations
        relations = self.relation_extractor.extract_relations(entities, text, paper_id)

        # Convert to triples
        triples = []

        for relation in relations:
            triple = Triple(
                subject=relation.subject.text,
                predicate=relation.predicate,
                object=relation.object.text,
                confidence=relation.confidence,
                provenance={
                    'paper_id': paper_id,
                    'evidence': relation.evidence,
                    'subject_type': relation.subject.type,
                    'object_type': relation.object.type,
                    'metadata': metadata or {}
                }
            )
            triples.append(triple)

        # Add entity type triples
        for entity in entities:
            triple = Triple(
                subject=entity.text,
                predicate='has_type',
                object=entity.type,
                confidence=entity.confidence,
                provenance={
                    'paper_id': paper_id,
                    'context': entity.context,
                    'metadata': metadata or {}
                }
            )
            triples.append(triple)

        # Add paper metadata triples
        if metadata:
            triples.extend(self._extract_metadata_triples(paper_id, metadata))

        return triples

    def _extract_metadata_triples(self, paper_id: str,
                                 metadata: Dict[str, Any]) -> List[Triple]:
        """Extract triples from paper metadata."""
        triples = []

        # Title triple
        if 'title' in metadata:
            triples.append(Triple(
                subject=paper_id,
                predicate='has_title',
                object=metadata['title'],
                confidence=1.0,
                provenance={'source': 'metadata'}
            ))

        # Author triples
        if 'authors' in metadata:
            for author in metadata['authors']:
                triples.append(Triple(
                    subject=paper_id,
                    predicate='has_author',
                    object=author,
                    confidence=1.0,
                    provenance={'source': 'metadata'}
                ))

        # Year triple
        if 'year' in metadata:
            triples.append(Triple(
                subject=paper_id,
                predicate='published_in',
                object=str(metadata['year']),
                confidence=1.0,
                provenance={'source': 'metadata'}
            ))

        # Venue triple
        if 'venue' in metadata:
            triples.append(Triple(
                subject=paper_id,
                predicate='published_at',
                object=metadata['venue'],
                confidence=1.0,
                provenance={'source': 'metadata'}
            ))

        return triples


class PaperProcessor:
    """Process scientific papers for knowledge extraction."""

    def __init__(self):
        """Initialize paper processor."""
        self.triple_extractor = TripleExtractor()
        self.processed_papers = set()

    def process_paper(self, paper_content: str, paper_id: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a scientific paper.

        Args:
            paper_content: Paper text content
            paper_id: Unique paper ID
            metadata: Paper metadata

        Returns:
            Processing results
        """
        if paper_id in self.processed_papers:
            logger.warning(f"Paper {paper_id} already processed")
            return {'status': 'already_processed', 'paper_id': paper_id}

        # Extract sections
        sections = self._extract_sections(paper_content)

        # Extract triples from each section
        all_triples = []

        for section_name, section_text in sections.items():
            triples = self.triple_extractor.extract_triples(
                section_text,
                paper_id,
                metadata
            )

            # Add section context
            for triple in triples:
                triple.provenance['section'] = section_name

            all_triples.extend(triples)

        # Mark as processed
        self.processed_papers.add(paper_id)

        # Create summary
        summary = self._create_extraction_summary(all_triples, paper_id, metadata)

        return {
            'status': 'success',
            'paper_id': paper_id,
            'triples': all_triples,
            'summary': summary,
            'metadata': metadata
        }

    def _extract_sections(self, paper_content: str) -> Dict[str, str]:
        """Extract sections from paper content."""
        sections = {
            'full_text': paper_content
        }

        # Common section headers
        section_patterns = [
            r'(?:^|\n)(?:Abstract|ABSTRACT)(?:\n|:)',
            r'(?:^|\n)(?:Introduction|INTRODUCTION)(?:\n|:)',
            r'(?:^|\n)(?:Related Work|RELATED WORK)(?:\n|:)',
            r'(?:^|\n)(?:Method|Methods|Methodology|METHODS?)(?:\n|:)',
            r'(?:^|\n)(?:Experiments?|EXPERIMENTS?)(?:\n|:)',
            r'(?:^|\n)(?:Results?|RESULTS?)(?:\n|:)',
            r'(?:^|\n)(?:Discussion|DISCUSSION)(?:\n|:)',
            r'(?:^|\n)(?:Conclusion|CONCLUSION)(?:\n|:)',
        ]

        # Extract each section
        for pattern_str in section_patterns:
            pattern = re.compile(pattern_str, re.M)
            matches = list(pattern.finditer(paper_content))

            if matches:
                section_name = matches[0].group().strip().lower().replace(':', '')
                start = matches[0].end()

                # Find next section or end of text
                next_section = None
                for next_pattern_str in section_patterns:
                    next_pattern = re.compile(next_pattern_str, re.M)
                    next_matches = list(next_pattern.finditer(paper_content[start:]))
                    if next_matches:
                        if next_section is None or next_matches[0].start() < next_section:
                            next_section = next_matches[0].start()

                if next_section:
                    section_text = paper_content[start:start + next_section]
                else:
                    section_text = paper_content[start:]

                sections[section_name] = section_text.strip()

        return sections

    def _create_extraction_summary(self, triples: List[Triple],
                                  paper_id: str,
                                  metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of extraction results."""
        # Count entities by type
        entity_counts = defaultdict(int)
        relation_counts = defaultdict(int)

        for triple in triples:
            if triple.predicate == 'has_type':
                entity_counts[triple.object] += 1
            else:
                relation_counts[triple.predicate] += 1

        # Get unique entities and relations
        unique_entities = set()
        unique_relations = set()

        for triple in triples:
            unique_entities.add(triple.subject)
            unique_entities.add(triple.object)
            unique_relations.add(triple.predicate)

        return {
            'paper_id': paper_id,
            'title': metadata.get('title', 'Unknown') if metadata else 'Unknown',
            'num_triples': len(triples),
            'num_unique_entities': len(unique_entities),
            'num_unique_relations': len(unique_relations),
            'entity_counts': dict(entity_counts),
            'relation_counts': dict(relation_counts),
            'avg_confidence': np.mean([t.confidence for t in triples]) if triples else 0
        }

    def batch_process(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple papers in batch.

        Args:
            papers: List of paper dicts with 'content', 'id', and optional 'metadata'

        Returns:
            List of processing results
        """
        results = []

        for paper in papers:
            result = self.process_paper(
                paper['content'],
                paper['id'],
                paper.get('metadata')
            )
            results.append(result)

        return results