# check_running_server.py - Check what's actually running
"""
Compare what should be running vs what is running
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def check_server_routes():
    """Check what routes the running server actually has"""
    print("🔍 Checking Running Server Routes")
    print("=" * 50)
    
    try:
        # Get OpenAPI spec from running server
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            
            print("✅ Server OpenAPI spec retrieved")
            
            # Extract all paths
            paths = openapi_data.get('paths', {})
            print(f"📊 Total endpoints found: {len(paths)}")
            
            auth_paths = [path for path in paths.keys() if path.startswith('/auth')]
            print(f"🔐 Auth endpoints found: {len(auth_paths)}")
            
            if auth_paths:
                print("   Auth endpoints:")
                for path in sorted(auth_paths):
                    methods = list(paths[path].keys())
                    print(f"   {methods} {path}")
            else:
                print("   ❌ NO AUTH ENDPOINTS FOUND")
            
            print("\n📋 All available endpoints:")
            for path in sorted(paths.keys()):
                methods = list(paths[path].keys())
                print(f"   {methods} {path}")
                
        else:
            print(f"❌ Failed to get OpenAPI spec: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking server routes: {e}")

def check_server_startup_logs():
    """Instructions for checking server startup"""
    print("\n" + "=" * 50)
    print("🔧 Debugging Steps:")
    print("1. Stop your current server (Ctrl+C)")
    print("2. Add debug prints to main_clean.py:")
    print("   Add this right after the auth_router import:")
    print()
    print("   # Debug auth router inclusion")
    print("   print(f'Auth router imported: {auth_router}')")
    print("   print(f'Auth router routes: {len(auth_router.routes)}')")
    print("   app.include_router(auth_router)")
    print("   print('Auth router included successfully')")
    print()
    print("3. Restart server and watch for those debug messages")
    print("4. If messages appear but endpoints still missing, there may be")
    print("   multiple FastAPI apps or import/execution order issues")

def test_direct_endpoint():
    """Test if endpoints exist but aren't in OpenAPI spec"""
    print("\n🧪 Testing Direct Endpoint Access")
    print("-" * 30)
    
    # Test endpoints that should exist
    test_endpoints = [
        "/auth/login",
        "/auth/debug/users",
        "/auth/profile"
    ]
    
    for endpoint in test_endpoints:
        try:
            # Try OPTIONS request to see if endpoint exists
            response = requests.options(f"{BASE_URL}{endpoint}")
            print(f"{endpoint}: {response.status_code}")
            
            if response.status_code == 405:  # Method not allowed = endpoint exists
                print(f"   ✅ Endpoint exists but doesn't accept OPTIONS")
            elif response.status_code == 404:
                print(f"   ❌ Endpoint not found")
            else:
                print(f"   ❓ Unexpected response: {response.status_code}")
                
        except Exception as e:
            print(f"{endpoint}: Error - {e}")

def main():
    check_server_routes()
    test_direct_endpoint()
    check_server_startup_logs()
    
    print("\n" + "=" * 50)
    print("🎯 Most Likely Issues:")
    print("1. Server cache - try restarting the server completely")
    print("2. Import order - auth_router import happens after app.include_router()")
    print("3. Exception during router inclusion that's being caught silently")
    print("4. Multiple app instances - check if you have multiple FastAPI() calls")

if __name__ == "__main__":
    main()