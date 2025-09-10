# debug_imports.py - Check what's causing import issues
"""
Debug script to identify import problems
"""

print("Testing imports one by one...")

# Test basic imports
try:
    from fastapi import FastAPI
    print("✅ FastAPI import works")
except Exception as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    from pydantic import BaseModel
    print("✅ Pydantic import works")
except Exception as e:
    print(f"❌ Pydantic import failed: {e}")

# Test utils imports individually
print("\nTesting utils imports...")

try:
    from utils.pagination import PaginationParams
    print("✅ Pagination import works")
except Exception as e:
    print(f"❌ Pagination import failed: {e}")

try:
    from utils.validation import FinancialValidator
    print("✅ Validation import works")
except Exception as e:
    print(f"❌ Validation import failed: {e}")

try:
    from utils.monitoring import rate_limit
    print("✅ Monitoring import works")
except Exception as e:
    print(f"❌ Monitoring import failed: {e}")

# Test psutil specifically
try:
    import psutil
    print("✅ psutil is available")
except ImportError:
    print("❌ psutil not installed - installing it will fix monitoring")

# Test if logs directory exists
import os
if os.path.exists('logs'):
    print("✅ logs directory exists")
else:
    print("❌ logs directory missing - creating it will fix logging")

print("\nTesting main_clean import...")
try:
    from main_clean import app
    print("✅ main_clean imports successfully")
except Exception as e:
    print(f"❌ main_clean import failed: {e}")
    print("This is the error we need to fix.")