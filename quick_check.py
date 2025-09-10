# quick_check.py
import subprocess
import sys

def get_installed_packages():
    result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                          capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')[2:]  # Skip header
    
    packages = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            name, version = parts[0], parts[1]
            packages[name] = version
    
    return packages

# Core packages your platform likely uses
core_packages = [
    'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy', 'alembic', 
    'PyJWT', 'passlib', 'bcrypt', 'pandas', 'numpy', 'scipy', 
    'requests', 'aiohttp', 'websockets', 'yfinance', 'python-dotenv'
]

print("Checking installed core packages:")
installed = get_installed_packages()

with open("requirements.txt", "w") as f:
    f.write("# Financial Platform Requirements\n\n")
    
    for pkg in core_packages:
        if pkg in installed:
            version = installed[pkg]
            f.write(f"{pkg}=={version}\n")
            print(f"✓ {pkg}=={version}")
        else:
            print(f"✗ {pkg} not installed")

print("\nGenerated basic requirements.txt")