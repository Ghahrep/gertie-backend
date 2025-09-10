# update_requirements.py
import subprocess
import sys

def get_version(package):
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", package], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        return None

# All packages your platform uses
all_packages = [
    'fastapi', 'uvicorn', 'pydantic', 'python-multipart',
    'sqlalchemy', 'alembic', 'PyJWT', 'passlib', 'bcrypt', 
    'pandas', 'numpy', 'scipy', 'requests', 'aiohttp', 
    'websockets', 'yfinance', 'python-dotenv',
    'PyPortfolioOpt', 'backtesting', 'hmmlearn', 'fbm', 'anthropic'
]

with open("requirements.txt", "w") as f:
    f.write("# Financial Platform Backend Requirements\n\n")
    
    for pkg in all_packages:
        version = get_version(pkg)
        if version:
            f.write(f"{pkg}=={version}\n")
            print(f"✓ {pkg}=={version}")
        else:
            f.write(f"# {pkg}  # Not installed\n")
            print(f"✗ {pkg} not installed")

print("\nUpdated requirements.txt with all packages")