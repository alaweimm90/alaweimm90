"""
Workflow Template System for ORCHEX
Provides reusable, parameterized workflow templates with validation and versioning.
"""

import os
import yaml
import json
import copy
import hashlib
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import re


class TemplateVersion(Enum):
    """Template versioning scheme."""
    V1_0 = "1.0.0"
    V1_1 = "1.1.0"
    V2_0 = "2.0.0"
    LATEST = "latest"


@dataclass
class TemplateParameter:
    """Represents a template parameter that can be filled in."""
    name: str
    type: str  # string, int, float, bool, list, dict
    description: str
    required: bool = True
    default: Any = None
    validation_pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate parameter value."""
        # Check type
        type_map = {
            'string': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict
        }

        expected_type = type_map.get(self.type)
        if expected_type and not isinstance(value, expected_type):
            try:
                # Try type coercion
                value = expected_type(value)
            except (ValueError, TypeError):
                return False, f"Parameter {self.name} must be of type {self.type}"

        # Check validation pattern
        if self.validation_pattern and self.type == 'string':
            if not re.match(self.validation_pattern, str(value)):
                return False, f"Parameter {self.name} does not match pattern {self.validation_pattern}"

        # Check allowed values
        if self.allowed_values and value not in self.allowed_values:
            return False, f"Parameter {self.name} must be one of {self.allowed_values}"

        return True, None


@dataclass
class WorkflowTemplate:
    """Base class for workflow templates."""

    name: str
    version: str
    category: str
    description: str
    author: str = "ORCHEX Team"
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, TemplateParameter] = field(default_factory=dict)
    workflow_definition: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize template ID."""
        self.template_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique template ID."""
        content = f"{self.name}:{self.version}:{self.category}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def add_parameter(self, param: TemplateParameter) -> None:
        """Add a parameter to the template."""
        self.parameters[param.name] = param

    def fill_parameters(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Fill template parameters with provided values."""
        # Validate all required parameters are provided
        missing = []
        for param_name, param in self.parameters.items():
            if param.required and param_name not in values:
                if param.default is None:
                    missing.append(param_name)
                else:
                    values[param_name] = param.default

        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Validate parameter values
        for param_name, value in values.items():
            if param_name in self.parameters:
                valid, error = self.parameters[param_name].validate(value)
                if not valid:
                    raise ValueError(f"Validation error: {error}")

        # Fill workflow definition with parameters
        filled_workflow = self._recursive_fill(
            copy.deepcopy(self.workflow_definition),
            values
        )

        return filled_workflow

    def _recursive_fill(self, obj: Any, values: Dict[str, Any]) -> Any:
        """Recursively fill template variables in object."""
        if isinstance(obj, str):
            # Replace template variables {{param_name}}
            for param_name, value in values.items():
                placeholder = f"{{{{{param_name}}}}}"
                if placeholder in obj:
                    obj = obj.replace(placeholder, str(value))
            return obj
        elif isinstance(obj, dict):
            return {k: self._recursive_fill(v, values) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_fill(item, values) for item in obj]
        else:
            return obj

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate template structure."""
        errors = []

        # Check required fields
        if not self.name:
            errors.append("Template name is required")
        if not self.version:
            errors.append("Template version is required")
        if not self.workflow_definition:
            errors.append("Workflow definition is required")

        # Check workflow definition structure
        if 'tasks' not in self.workflow_definition:
            errors.append("Workflow definition must contain 'tasks'")

        # Validate parameter references
        template_vars = self._extract_template_vars(self.workflow_definition)
        undefined_vars = template_vars - set(self.parameters.keys())
        if undefined_vars:
            errors.append(f"Undefined template variables: {undefined_vars}")

        return len(errors) == 0, errors

    def _extract_template_vars(self, obj: Any, vars_set: Optional[Set[str]] = None) -> Set[str]:
        """Extract all template variable names from object."""
        if vars_set is None:
            vars_set = set()

        if isinstance(obj, str):
            # Find all {{param_name}} patterns
            matches = re.findall(r'\{\{(\w+)\}\}', obj)
            vars_set.update(matches)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._extract_template_vars(value, vars_set)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_template_vars(item, vars_set)

        return vars_set

    def to_yaml(self) -> str:
        """Export template to YAML format."""
        export_data = {
            'name': self.name,
            'version': self.version,
            'category': self.category,
            'description': self.description,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'parameters': {
                name: {
                    'type': param.type,
                    'description': param.description,
                    'required': param.required,
                    'default': param.default,
                    'validation_pattern': param.validation_pattern,
                    'allowed_values': param.allowed_values
                }
                for name, param in self.parameters.items()
            },
            'workflow': self.workflow_definition,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }
        return yaml.dump(export_data, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'WorkflowTemplate':
        """Create template from YAML content."""
        data = yaml.safe_load(yaml_content)

        # Parse parameters
        parameters = {}
        for name, param_data in data.get('parameters', {}).items():
            parameters[name] = TemplateParameter(
                name=name,
                type=param_data['type'],
                description=param_data['description'],
                required=param_data.get('required', True),
                default=param_data.get('default'),
                validation_pattern=param_data.get('validation_pattern'),
                allowed_values=param_data.get('allowed_values')
            )

        # Create template
        template = cls(
            name=data['name'],
            version=data['version'],
            category=data['category'],
            description=data['description'],
            author=data.get('author', 'ORCHEX Team'),
            tags=data.get('tags', []),
            parameters=parameters,
            workflow_definition=data['workflow'],
            dependencies=data.get('dependencies', []),
            metadata=data.get('metadata', {})
        )

        # Parse dates if present
        if 'created_at' in data:
            template.created_at = datetime.datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            template.updated_at = datetime.datetime.fromisoformat(data['updated_at'])

        return template

    def compose_with(self, other: 'WorkflowTemplate',
                    connection_type: str = 'sequential') -> 'WorkflowTemplate':
        """Compose this template with another template."""
        if connection_type == 'sequential':
            # Connect templates sequentially
            composed_workflow = {
                'tasks': []
            }

            # Add tasks from first template
            for task in self.workflow_definition.get('tasks', []):
                new_task = copy.deepcopy(task)
                new_task['name'] = f"{self.name}_{task.get('name', 'task')}"
                composed_workflow['tasks'].append(new_task)

            # Add tasks from second template with dependency
            last_task = composed_workflow['tasks'][-1]['name'] if composed_workflow['tasks'] else None
            for task in other.workflow_definition.get('tasks', []):
                new_task = copy.deepcopy(task)
                new_task['name'] = f"{other.name}_{task.get('name', 'task')}"
                if last_task and 'depends_on' not in new_task:
                    new_task['depends_on'] = [last_task]
                composed_workflow['tasks'].append(new_task)

        elif connection_type == 'parallel':
            # Run templates in parallel
            composed_workflow = {
                'tasks': [
                    {
                        'name': f"{self.name}_group",
                        'type': 'parallel',
                        'subtasks': self.workflow_definition.get('tasks', [])
                    },
                    {
                        'name': f"{other.name}_group",
                        'type': 'parallel',
                        'subtasks': other.workflow_definition.get('tasks', [])
                    }
                ]
            }
        else:
            raise ValueError(f"Unknown connection type: {connection_type}")

        # Merge parameters
        combined_params = {**self.parameters, **other.parameters}

        # Create composed template
        composed = WorkflowTemplate(
            name=f"{self.name}_composed_{other.name}",
            version="1.0.0",
            category="composed",
            description=f"Composition of {self.name} and {other.name}",
            parameters=combined_params,
            workflow_definition=composed_workflow,
            dependencies=list(set(self.dependencies + other.dependencies)),
            tags=list(set(self.tags + other.tags))
        )

        return composed


class TemplateRegistry:
    """Registry for managing workflow templates."""

    def __init__(self, template_dir: Optional[str] = None):
        """Initialize template registry."""
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), '..', 'templates'
        )
        self.categories: Dict[str, List[str]] = {}
        self.tags_index: Dict[str, List[str]] = {}

        # Create template directory if it doesn't exist
        Path(self.template_dir).mkdir(parents=True, exist_ok=True)

        # Load existing templates
        self.discover_templates()

    def register(self, template: WorkflowTemplate) -> None:
        """Register a template."""
        # Validate template
        valid, errors = template.validate()
        if not valid:
            raise ValueError(f"Template validation failed: {errors}")

        # Add to registry
        self.templates[template.template_id] = template

        # Update indices
        if template.category not in self.categories:
            self.categories[template.category] = []
        self.categories[template.category].append(template.template_id)

        for tag in template.tags:
            if tag not in self.tags_index:
                self.tags_index[tag] = []
            self.tags_index[tag].append(template.template_id)

    def get(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)

    def get_by_name(self, name: str, version: str = "latest") -> Optional[WorkflowTemplate]:
        """Get template by name and version."""
        for template in self.templates.values():
            if template.name == name:
                if version == "latest" or template.version == version:
                    return template
        return None

    def list_by_category(self, category: str) -> List[WorkflowTemplate]:
        """List templates by category."""
        template_ids = self.categories.get(category, [])
        return [self.templates[tid] for tid in template_ids]

    def list_by_tag(self, tag: str) -> List[WorkflowTemplate]:
        """List templates by tag."""
        template_ids = self.tags_index.get(tag, [])
        return [self.templates[tid] for tid in template_ids]

    def search(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name or description."""
        query_lower = query.lower()
        results = []

        for template in self.templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)

        return results

    def discover_templates(self) -> None:
        """Discover and load templates from template directory."""
        if not os.path.exists(self.template_dir):
            return

        for file_path in Path(self.template_dir).glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    template = WorkflowTemplate.from_yaml(f.read())
                    self.register(template)
            except Exception as e:
                print(f"Error loading template {file_path}: {e}")

    def save_template(self, template: WorkflowTemplate) -> None:
        """Save template to file."""
        # Register template
        self.register(template)

        # Save to file
        filename = f"{template.name}_{template.version}.yaml".replace('.', '_')
        file_path = os.path.join(self.template_dir, filename)

        with open(file_path, 'w') as f:
            f.write(template.to_yaml())

    def export_catalog(self) -> Dict[str, Any]:
        """Export template catalog."""
        catalog = {
            'total_templates': len(self.templates),
            'categories': {},
            'templates': []
        }

        for category, template_ids in self.categories.items():
            catalog['categories'][category] = len(template_ids)

        for template in self.templates.values():
            catalog['templates'].append({
                'id': template.template_id,
                'name': template.name,
                'version': template.version,
                'category': template.category,
                'description': template.description,
                'tags': template.tags,
                'parameters_count': len(template.parameters),
                'created_at': template.created_at.isoformat()
            })

        return catalog


class TemplateBuilder:
    """Builder class for creating templates programmatically."""

    def __init__(self, name: str):
        """Initialize template builder."""
        self.template = WorkflowTemplate(
            name=name,
            version="1.0.0",
            category="custom",
            description=""
        )
        self.current_task_index = 0

    def with_version(self, version: str) -> 'TemplateBuilder':
        """Set template version."""
        self.template.version = version
        return self

    def with_category(self, category: str) -> 'TemplateBuilder':
        """Set template category."""
        self.template.category = category
        return self

    def with_description(self, description: str) -> 'TemplateBuilder':
        """Set template description."""
        self.template.description = description
        return self

    def with_author(self, author: str) -> 'TemplateBuilder':
        """Set template author."""
        self.template.author = author
        return self

    def with_tags(self, *tags: str) -> 'TemplateBuilder':
        """Add tags to template."""
        self.template.tags.extend(tags)
        return self

    def add_parameter(self, name: str, param_type: str, description: str,
                     required: bool = True, default: Any = None,
                     validation_pattern: Optional[str] = None,
                     allowed_values: Optional[List[Any]] = None) -> 'TemplateBuilder':
        """Add parameter to template."""
        param = TemplateParameter(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default=default,
            validation_pattern=validation_pattern,
            allowed_values=allowed_values
        )
        self.template.add_parameter(param)
        return self

    def add_task(self, name: str, task_type: str, **kwargs) -> 'TemplateBuilder':
        """Add task to workflow."""
        if 'tasks' not in self.template.workflow_definition:
            self.template.workflow_definition['tasks'] = []

        task = {
            'name': name,
            'type': task_type,
            **kwargs
        }

        self.template.workflow_definition['tasks'].append(task)
        self.current_task_index += 1
        return self

    def add_parallel_tasks(self, *tasks: Dict[str, Any]) -> 'TemplateBuilder':
        """Add parallel tasks to workflow."""
        if 'tasks' not in self.template.workflow_definition:
            self.template.workflow_definition['tasks'] = []

        parallel_group = {
            'name': f"parallel_group_{self.current_task_index}",
            'type': 'parallel',
            'subtasks': list(tasks)
        }

        self.template.workflow_definition['tasks'].append(parallel_group)
        self.current_task_index += 1
        return self

    def add_conditional(self, condition: str, if_tasks: List[Dict],
                       else_tasks: Optional[List[Dict]] = None) -> 'TemplateBuilder':
        """Add conditional branching."""
        if 'tasks' not in self.template.workflow_definition:
            self.template.workflow_definition['tasks'] = []

        conditional = {
            'name': f"conditional_{self.current_task_index}",
            'type': 'conditional',
            'condition': condition,
            'if_branch': if_tasks
        }

        if else_tasks:
            conditional['else_branch'] = else_tasks

        self.template.workflow_definition['tasks'].append(conditional)
        self.current_task_index += 1
        return self

    def add_loop(self, items: str, loop_tasks: List[Dict]) -> 'TemplateBuilder':
        """Add loop over items."""
        if 'tasks' not in self.template.workflow_definition:
            self.template.workflow_definition['tasks'] = []

        loop = {
            'name': f"loop_{self.current_task_index}",
            'type': 'loop',
            'items': items,
            'tasks': loop_tasks
        }

        self.template.workflow_definition['tasks'].append(loop)
        self.current_task_index += 1
        return self

    def with_dependency(self, *dependencies: str) -> 'TemplateBuilder':
        """Add template dependencies."""
        self.template.dependencies.extend(dependencies)
        return self

    def with_metadata(self, **metadata: Any) -> 'TemplateBuilder':
        """Add metadata to template."""
        self.template.metadata.update(metadata)
        return self

    def build(self) -> WorkflowTemplate:
        """Build and validate template."""
        valid, errors = self.template.validate()
        if not valid:
            raise ValueError(f"Template validation failed: {errors}")
        return self.template


# Utility functions
def create_template_from_workflow(workflow: Dict[str, Any],
                                 name: str,
                                 extract_params: bool = True) -> WorkflowTemplate:
    """Create template from existing workflow."""
    builder = TemplateBuilder(name)

    if extract_params:
        # Extract template variables from workflow
        def extract_vars(obj):
            vars_found = set()
            if isinstance(obj, str):
                matches = re.findall(r'\{\{(\w+)\}\}', obj)
                vars_found.update(matches)
            elif isinstance(obj, dict):
                for value in obj.values():
                    vars_found.update(extract_vars(value))
            elif isinstance(obj, list):
                for item in obj:
                    vars_found.update(extract_vars(item))
            return vars_found

        template_vars = extract_vars(workflow)

        # Add parameters for each variable
        for var in template_vars:
            builder.add_parameter(
                name=var,
                param_type='string',
                description=f"Parameter {var}",
                required=True
            )

    # Set workflow definition
    builder.template.workflow_definition = workflow

    return builder.build()


def merge_templates(*templates: WorkflowTemplate,
                   merge_strategy: str = 'sequential') -> WorkflowTemplate:
    """Merge multiple templates into one."""
    if not templates:
        raise ValueError("At least one template required")

    if len(templates) == 1:
        return templates[0]

    # Start with first template
    result = templates[0]

    # Compose with remaining templates
    for template in templates[1:]:
        result = result.compose_with(template, merge_strategy)

    return result