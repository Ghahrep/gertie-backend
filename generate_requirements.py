# generate_requirements.py
"""
Generate requirements.txt from project imports - Improved Version
================================================================

Scans Python files to identify imported packages and generates requirements.txt
with proper filtering and error handling.
"""

import ast
import os
import sys
import re
from pathlib import Path
from typing import Set, Dict, List, Optional

def get_installed_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package with error handling"""
    try:
        # Skip invalid package names
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9\-_.]*$', package_name):
            return None
        
        import pkg_resources
        return pkg_resources.get_distribution(package_name).version
    except (pkg_resources.DistributionNotFound, 
            pkg_resources.extern.packaging.requirements.InvalidRequirement, 
            Exception):
        return None

def extract_imports_from_file(file_path: str) -> Set[str]:
    """Extract imports from a Python file with better filtering"""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if is_valid_package_name(module_name):
                        imports.add(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if is_valid_package_name(module_name):
                        imports.add(module_name)
                        
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
    
    return imports

def is_valid_package_name(name: str) -> bool:
    """Check if a name could be a valid Python package"""
    if not name or len(name) < 2:
        return False
    
    # Must start with letter or underscore, contain only valid chars
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        return False
    
    # Skip common false positives
    false_positives = {
        '__future__', '__main__', '__name__', '__file__', '__doc__',
        '__builtins__', '__package__', '__spec__', '__loader__',
        'test', 'tests', 'example', 'examples', 'demo', 'sample'
    }
    
    return name not in false_positives

def get_comprehensive_stdlib_modules() -> Set[str]:
    """Get comprehensive list of standard library modules"""
    # Python standard library modules (comprehensive list)
    stdlib_modules = {
        # Core built-ins and system
        'os', 'sys', 'json', 'datetime', 'time', 'random', 'math', 'cmath',
        'collections', 'itertools', 'functools', 'operator', 'copy', 'pickle',
        'csv', 'sqlite3', 'urllib', 'http', 'email', 'html', 'xml', 'logging',
        'warnings', 'threading', 'multiprocessing', 'concurrent', 'asyncio',
        'socket', 'ssl', 'hashlib', 'hmac', 'secrets', 'uuid', 'base64',
        'binascii', 'struct', 'codecs', 'io', 'stringio', 'textwrap', 're',
        'difflib', 'unicodedata', 'locale', 'calendar', 'zoneinfo', 'decimal',
        'fractions', 'statistics', 'pathlib', 'glob', 'tempfile', 'shutil',
        'gzip', 'zipfile', 'tarfile', 'configparser', 'argparse', 'getopt',
        'pdb', 'trace', 'traceback', 'inspect', 'dis', 'types', 'typing',
        'abc', 'contextlib', 'weakref', 'gc', 'ctypes', 'platform', 'keyword',
        'token', 'tokenize', 'ast', 'symtable', 'symbol', 'compileall',
        'py_compile', 'zipimport', 'pkgutil', 'modulefinder', 'runpy',
        'importlib', 'imp', 'formatter', 'errno', 'faulthandler', 'tracemalloc',
        
        # Data processing and formats
        'heapq', 'bisect', 'array', 'enum', 'dataclasses', 'reprlib', 'pprint',
        'string', 'textwrap', 'filecmp', 'fnmatch', 'linecache', 'shlex',
        
        # Development and debugging  
        'unittest', 'doctest', 'test', 'timeit', 'cProfile', 'profile',
        'pstats', 'hotshot', 'tabnanny', 'compileall',
        
        # Network and internet
        'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib', 'telnetlib',
        'xmlrpc', 'webbrowser', 'cgi', 'cgitb', 'wsgiref', 'urllib2',
        'urlparse', 'cookielib', 'Cookie', 'BaseHTTPServer', 'SimpleHTTPServer',
        'CGIHTTPServer', 'SocketServer',
        
        # GUI (tkinter)
        'tkinter', 'Tkinter', 'turtle', 'turtledemo',
        
        # Platform-specific
        'posix', 'pwd', 'spwd', 'grp', 'crypt', 'dl', 'termios', 'tty',
        'pty', 'fcntl', 'pipes', 'resource', 'nis', 'syslog', 'mmap',
        'select', 'msvcrt', 'winreg', '_winreg', 'winsound',
        
        # Deprecated/legacy modules
        'imp', 'formatter', 'ihooks', 'fpformat', 'dircache', 'statcache',
        'fileinput', 'user', 'new', 'rexec', 'Bastion', 'rfc822', 'mimetools',
        'MimeWriter', 'mimify', 'multifile', 'ConfigParser', 'Queue',
        'SocketServer', 'BaseHTTPServer', 'SimpleHTTPServer', 'CGIHTTPServer',
        'cookielib', 'Cookie', 'htmlentitydefs', 'HTMLParser', 'htmllib',
        'sgmllib', 'urlparse', 'urllib2', 'StringIO', 'cStringIO', 'UserDict',
        'UserList', 'UserString', 'commands', 'dl', 'fpformat', 'ihooks',
        'imageop', 'rgbimg', 'sha', 'md5',
        
        # Common internal modules
        'builtins', '_ast', '_codecs', '_collections', '_functools', '_io',
        '_json', '_locale', '_md5', '_operator', '_pickle', '_random', '_sha1',
        '_sha256', '_sha512', '_socket', '_sqlite3', '_ssl', '_string',
        '_struct', '_thread', '_threading_local', '_weakref', '_winapi'
    }
    
    return stdlib_modules

def scan_project_for_actual_imports(project_root: str = ".") -> Set[str]:
    """Scan project files more selectively"""
    all_imports = set()
    
    # Key files to scan
    important_files = [
        "main_clean.py",
        "audit_tools.py", 
        "test_websocket_basic.py",
        "simple_test_runner.py"
    ]
    
    # Key directories
    important_dirs = [
        "auth",
        "db", 
        "services",
        "tools",
        "websocket",
        "utils"
    ]
    
    print("Scanning key project files...")
    
    # Scan important files
    for file_name in important_files:
        file_path = os.path.join(project_root, file_name)
        if os.path.exists(file_path):
            print(f"  Scanning: {file_name}")
            imports = extract_imports_from_file(file_path)
            all_imports.update(imports)
    
    # Scan important directories
    for dir_name in important_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            for file_path in Path(dir_path).glob("**/*.py"):
                if file_path.name != "__pycache__":
                    relative_path = os.path.relpath(file_path, project_root)
                    print(f"  Scanning: {relative_path}")
                    imports = extract_imports_from_file(str(file_path))
                    all_imports.update(imports)
    
    return all_imports

def filter_to_actual_packages(imports: Set[str]) -> Set[str]:
    """Filter to only actual third-party packages"""
    stdlib = get_comprehensive_stdlib_modules()
    
    # Remove standard library
    filtered = {imp for imp in imports if imp not in stdlib}
    
    # Additional filtering for known patterns
    filtered = {imp for imp in filtered if not imp.startswith('_')}
    filtered = {imp for imp in filtered if len(imp) > 1}
    
    return filtered

def get_known_package_mapping() -> Dict[str, str]:
    """Map import names to actual package names"""
    return {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python', 
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'jwt': 'PyJWT',
        'dateutil': 'python-dateutil',
        'psycopg2': 'psycopg2-binary',
        'MySQLdb': 'mysqlclient',
        'bs4': 'beautifulsoup4',
        'requests_oauthlib': 'requests-oauthlib',
        'google': 'google-api-python-client'
    }

def generate_clean_requirements(packages_with_versions: Dict[str, str]) -> str:
    """Generate clean requirements.txt content"""
    
    # Core categories for your financial platform
    categories = {
        'Web Framework': [
            'fastapi', 'uvicorn', 'pydantic', 'starlette', 'python-multipart'
        ],
        'Database': [
            'sqlalchemy', 'alembic', 'psycopg2-binary', 'psycopg2', 'pymysql'
        ],
        'Authentication': [
            'PyJWT', 'passlib', 'bcrypt', 'python-jose'
        ],
        'Financial Data & Analysis': [
            'yfinance', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
            'PyPortfolioOpt', 'backtesting', 'arch', 'statsmodels'
        ],
        'Machine Learning': [
            'scikit-learn', 'hmmlearn', 'fbm'
        ],
        'HTTP & Requests': [
            'requests', 'aiohttp', 'httpx', 'urllib3'
        ],
        'WebSocket': [
            'websockets', 'python-socketio'
        ],
        'AI Integration': [
            'anthropic', 'openai', 'langchain'
        ],
        'Utilities': [
            'python-dotenv', 'tqdm', 'click', 'rich', 'python-dateutil'
        ]
    }
    
    content = [
        "# Financial Platform Backend - Requirements",
        "# Generated automatically on " + str(datetime.datetime.now().date()),
        "",
        "# Core dependencies for production deployment",
        ""
    ]
    
    # Add categorized packages
    all_categorized = set()
    for category_name, category_packages in categories.items():
        found_packages = []
        
        for package in category_packages:
            if package in packages_with_versions:
                version = packages_with_versions[package]
                found_packages.append(f"{package}=={version}")
                all_categorized.add(package)
        
        if found_packages:
            content.append(f"# {category_name}")
            content.extend(sorted(found_packages))
            content.append("")
    
    # Add remaining packages
    remaining = []
    for package, version in packages_with_versions.items():
        if package not in all_categorized:
            remaining.append(f"{package}=={version}")
    
    if remaining:
        content.append("# Additional Dependencies")
        content.extend(sorted(remaining))
        content.append("")
    
    return "\n".join(content)

def main():
    """Main execution function"""
    import datetime
    
    print("Financial Platform Requirements Generator (Improved)")
    print("=" * 55)
    
    # Scan for imports
    print("\n1. Scanning project files...")
    raw_imports = scan_project_for_actual_imports()
    print(f"   Found {len(raw_imports)} raw imports")
    
    # Filter to actual packages
    print("\n2. Filtering to third-party packages...")
    actual_packages = filter_to_actual_packages(raw_imports)
    print(f"   Found {len(actual_packages)} potential third-party packages")
    
    # Map to real package names and get versions
    print("\n3. Checking installed packages and versions...")
    package_mapping = get_known_package_mapping()
    packages_with_versions = {}
    packages_not_found = []
    
    for import_name in sorted(actual_packages):
        # Map to actual package name
        actual_package = package_mapping.get(import_name, import_name)
        
        # Get version
        version = get_installed_version(actual_package)
        
        if version:
            packages_with_versions[actual_package] = version
            print(f"   ✓ {actual_package:<20} v{version}")
        else:
            packages_not_found.append(actual_package)
            print(f"   ✗ {actual_package:<20} NOT INSTALLED")
    
    print(f"\n4. Summary:")
    print(f"   - Installed packages: {len(packages_with_versions)}")
    print(f"   - Missing packages: {len(packages_not_found)}")
    
    if packages_with_versions:
        print("\n5. Generating requirements.txt...")
        
        # Backup existing file
        if os.path.exists("requirements.txt"):
            import shutil
            shutil.copy2("requirements.txt", "requirements.txt.backup")
            print("   Backed up existing requirements.txt")
        
        # Generate new requirements
        requirements_content = generate_clean_requirements(packages_with_versions)
        
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        
        print("   ✓ Generated requirements.txt")
        
        # Show preview
        print("\n6. Preview of requirements.txt:")
        print("-" * 40)
        lines = requirements_content.split('\n')
        for line in lines[:25]:
            print(line)
        if len(lines) > 25:
            print(f"... and {len(lines) - 25} more lines")
        
        print(f"\n✓ Successfully generated requirements.txt with {len(packages_with_versions)} packages")
        print("\nNext steps:")
        print("1. Review the generated requirements.txt")
        print("2. Test with: pip install -r requirements.txt")
        print("3. Commit to version control")
        
    else:
        print("\n❌ No installed packages found to generate requirements.txt")

if __name__ == "__main__":
    main()