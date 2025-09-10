import os
import importlib
import inspect

def audit_tools():
    tools_dir = "tools"
    modules = []

    # Find all .py files in tools folder except __init__.py and __pycache__
    for file in os.listdir(tools_dir):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]  # remove .py
            modules.append(module_name)

    print(f"Found {len(modules)} tool modules: {modules}")

    for module_name in modules:
        try:
            module = importlib.import_module(f"{tools_dir}.{module_name}")
            print(f"\nModule: {module_name}")
            
            # Get all public functions
            functions = [name for name, obj in inspect.getmembers(module, inspect.isfunction)
                         if not name.startswith("_")]
            print(f"  {len(functions)} functions: {functions}")
        except Exception as e:
            print(f"  ⚠️ Could not load {module_name}: {e}")

if __name__ == "__main__":
    audit_tools()
