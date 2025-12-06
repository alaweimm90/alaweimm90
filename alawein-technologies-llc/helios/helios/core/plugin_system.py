"""
Plugin System - Framework for extending HELIOS

Features:
- Dynamic plugin discovery and loading
- Service registry for dependency injection
- Plugin versioning and compatibility checking
- Hook-based event system
- Plugin lifecycle management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional, Type
from pathlib import Path
import importlib.util
import sys
from enum import Enum
import json


class PluginType(str, Enum):
    """Plugin type enumeration"""
    DOMAIN = "domain"
    VALIDATOR = "validator"
    OPTIMIZER = "optimizer"
    EXPORTER = "exporter"
    CUSTOM = "custom"


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    plugin_type: PluginType
    author: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    min_helios_version: str = "0.1.0"

    def is_compatible(self, helios_version: str) -> bool:
        """Check if plugin is compatible with HELIOS version"""
        # Simple version comparison (x.y.z format)
        def parse_version(v: str) -> tuple:
            return tuple(map(int, v.split('.')))

        return parse_version(helios_version) >= parse_version(self.min_helios_version)


class PluginInterface(ABC):
    """Base interface for all plugins"""

    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.enabled = True

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate plugin can run"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup on shutdown"""
        pass


class ServiceRegistry:
    """Registry for services and dependencies"""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}

    def register(self, name: str, service: Any, singleton: bool = True) -> None:
        """Register a service instance"""
        self._services[name] = service
        if singleton:
            self._singletons[name] = service

    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory function"""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """Get service instance"""
        if name in self._singletons:
            return self._singletons[name]

        if name in self._factories:
            return self._factories[name]()

        if name in self._services:
            return self._services[name]

        raise ValueError(f"Service '{name}' not registered")

    def has(self, name: str) -> bool:
        """Check if service is registered"""
        return name in (self._services | self._factories | self._singletons)


class PluginManager:
    """Manage plugin lifecycle and execution"""

    def __init__(self, plugins_dir: Optional[Path] = None):
        self.plugins_dir = plugins_dir or Path("helios/plugins")
        self.plugins: Dict[str, PluginInterface] = {}
        self.service_registry = ServiceRegistry()
        self.hooks: Dict[str, List[Callable]] = {}

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback"""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)

    def trigger_hook(self, hook_name: str, *args, **kwargs) -> None:
        """Trigger all callbacks for a hook"""
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                callback(*args, **kwargs)

    def load_plugin(self, plugin_path: Path, config: Dict[str, Any] = None) -> None:
        """Load a plugin from file"""
        config = config or {}

        # Load metadata
        metadata_file = plugin_path / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"No metadata.json in {plugin_path}")

        with open(metadata_file) as f:
            metadata_dict = json.load(f)
        metadata = PluginMetadata(**metadata_dict)

        # Check compatibility
        if not metadata.is_compatible("0.1.0"):
            raise ValueError(
                f"Plugin {metadata.name} requires HELIOS >= {metadata.min_helios_version}"
            )

        # Load plugin module
        plugin_file = plugin_path / "plugin.py"
        spec = importlib.util.spec_from_file_location("plugin", plugin_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["plugin"] = module
        spec.loader.exec_module(module)

        # Get plugin class (assume it's named Plugin)
        plugin_class = getattr(module, "Plugin")
        plugin_instance = plugin_class(metadata)

        # Initialize
        plugin_instance.initialize(config)
        if not plugin_instance.validate():
            raise ValueError(f"Plugin {metadata.name} failed validation")

        self.plugins[metadata.name] = plugin_instance
        self.trigger_hook("plugin_loaded", metadata.name, plugin_instance)

    def load_plugins_from_dir(self, config: Dict[str, Any] = None) -> None:
        """Load all plugins from directory"""
        if not self.plugins_dir.exists():
            return

        config = config or {}
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir():
                try:
                    self.load_plugin(plugin_dir, config)
                except Exception as e:
                    print(f"⚠️ Failed to load plugin {plugin_dir.name}: {e}")

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of a type"""
        return [
            p for p in self.plugins.values()
            if p.metadata.plugin_type == plugin_type and p.enabled
        ]

    def shutdown_all(self) -> None:
        """Shutdown all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.shutdown()
            except Exception as e:
                print(f"⚠️ Error shutting down {plugin.metadata.name}: {e}")


class DependencyInjection:
    """Simple dependency injection container"""

    def __init__(self):
        self.container: Dict[str, Any] = {}
        self.factories: Dict[str, Callable] = {}

    def register_singleton(self, name: str, factory: Callable) -> None:
        """Register singleton with lazy initialization"""
        self.factories[name] = factory

    def resolve(self, name: str) -> Any:
        """Resolve dependency"""
        # Check if already created
        if name in self.container:
            return self.container[name]

        # Create if factory exists
        if name in self.factories:
            instance = self.factories[name](self)
            self.container[name] = instance
            return instance

        raise ValueError(f"Cannot resolve dependency: {name}")

    def inject(self, cls: Type):
        """Decorator for constructor injection"""
        def wrapper(*args, **kwargs):
            # Inspect constructor and inject dependencies
            import inspect
            sig = inspect.signature(cls.__init__)
            injected = {}

            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param_name in self.container or param_name in self.factories:
                    injected[param_name] = self.resolve(param_name)

            return cls(*args, **{**injected, **kwargs})

        return wrapper


# Global instances
_plugin_manager: Optional[PluginManager] = None
_service_registry: Optional[ServiceRegistry] = None
_di_container: Optional[DependencyInjection] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def get_service_registry() -> ServiceRegistry:
    """Get global service registry"""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry


def get_di_container() -> DependencyInjection:
    """Get global DI container"""
    global _di_container
    if _di_container is None:
        _di_container = DependencyInjection()
    return _di_container


def initialize_plugins(config: Dict[str, Any] = None) -> None:
    """Initialize plugin system"""
    config = config or {}
    manager = get_plugin_manager()
    manager.load_plugins_from_dir(config)
    print(f"✅ Loaded {len(manager.plugins)} plugins")
