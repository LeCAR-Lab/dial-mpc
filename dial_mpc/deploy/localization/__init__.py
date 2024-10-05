import threading
import pkgutil
import os
import importlib

plugin_registry = {}
_registry_lock = threading.Lock()

def get_available_plugins():
    with _registry_lock:
        return list(plugin_registry.keys())

def discover_builtin_plugins():
    plugin_path = os.path.dirname(__file__)
    for finder, name, ispkg in pkgutil.iter_modules([plugin_path]):
        if name not in plugin_registry and name != 'base_plugin':
            plugin_registry[name] = None  # Placeholder for lazy loading

discover_builtin_plugins()

def register_plugin(name, plugin_cls=None, module_path=None):
    with _registry_lock:
        if name in plugin_registry:
            raise ValueError(f"Plugin '{name}' is already registered.")

        if plugin_cls:
            # Ensure the plugin class is a subclass of BaseLocalizationPlugin
            from .base_plugin import BaseLocalizationPlugin
            if not issubclass(plugin_cls, BaseLocalizationPlugin):
                raise TypeError("The plugin class must inherit from BaseLocalizationPlugin.")
            plugin_registry[name] = plugin_cls

        elif module_path:
            # Dynamically load the module from the given path
            import importlib.util
            spec = importlib.util.spec_from_file_location(name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Expect the module to define 'Plugin' class
            plugin_cls = getattr(module, 'Plugin', None)
            if not plugin_cls:
                raise AttributeError(f"No 'Plugin' class found in '{module_path}'.")
            from .base_plugin import BaseLocalizationPlugin
            if not issubclass(plugin_cls, BaseLocalizationPlugin):
                raise TypeError("The plugin class must inherit from BaseLocalizationPlugin.")
            plugin_registry[name] = plugin_cls

        else:
            raise ValueError("You must provide either 'plugin_cls' or 'module_path'.")

def load_plugin(plugin_name):
    with _registry_lock:
        plugin_cls = plugin_registry.get(plugin_name)

        if plugin_cls is None:
            # Lazy loading of built-in plugins
            try:
                module = importlib.import_module(f".{plugin_name}", package=__package__)
                plugin_cls = getattr(module, 'Plugin', None)
                if not plugin_cls:
                    print(f"No 'Plugin' class in module '{plugin_name}'.")
                    return None
                from .base_plugin import BaseLocalizationPlugin
                if not issubclass(plugin_cls, BaseLocalizationPlugin):
                    print(f"'{plugin_name}' does not inherit from BaseLocalizationPlugin.")
                    return None
                plugin_registry[plugin_name] = plugin_cls
            except ImportError as e:
                print(f"Failed to import plugin '{plugin_name}': {e}")
                return None
        elif plugin_cls is not None:
            # Plugin class already loaded
            pass
        else:
            print(f"Plugin '{plugin_name}' is not registered.")
            return None

    # Instantiate and return the plugin
    try:
        return plugin_cls()
    except Exception as e:
        print(f"Failed to instantiate plugin '{plugin_name}': {e}")
        return None
