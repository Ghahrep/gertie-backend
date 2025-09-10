# check_auth_setup.py - Debug authentication endpoints
"""
Check if authentication endpoints are properly set up
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def check_available_endpoints():
    """Check what endpoints are available"""
    print("Checking available endpoints...")
    
    # Try to get OpenAPI spec
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ /docs endpoint available - check http://127.0.0.1:8000/docs for all endpoints")
        else:
            print(f"❌ /docs endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ /docs endpoint error: {e}")
    
    # Try OpenAPI JSON
    try:
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = list(openapi_data.get('paths', {}).keys())
            print(f"✅ Available endpoints from OpenAPI:")
            for path in sorted(paths):
                print(f"   {path}")
        else:
            print(f"❌ /openapi.json failed: {response.status_code}")
    except Exception as e:
        print(f"❌ OpenAPI error: {e}")

def test_auth_endpoints():
    """Test various auth endpoint possibilities"""
    print("\nTesting authentication endpoint variations...")
    
    auth_endpoints = [
        "/auth/login",
        "/login", 
        "/token",
        "/auth/token",
        "/api/auth/login"
    ]
    
    for endpoint in auth_endpoints:
        try:
            # Try GET first
            response = requests.get(f"{BASE_URL}{endpoint}")
            print(f"GET {endpoint}: {response.status_code}")
            
            # Try POST with form data
            login_data = {
                "username": "user1@example.com", 
                "password": "user123"
            }
            response = requests.post(f"{BASE_URL}{endpoint}", data=login_data)
            print(f"POST {endpoint} (form): {response.status_code}")
            
            # Try POST with JSON
            response = requests.post(f"{BASE_URL}{endpoint}", json=login_data)
            print(f"POST {endpoint} (json): {response.status_code}")
            
        except Exception as e:
            print(f"Error testing {endpoint}: {e}")

def check_auth_router_inclusion():
    """Check if auth router is properly included"""
    print("\nChecking if authentication router is included...")
    
    # This would normally check the main_clean.py file
    # but we'll check via the API instead
    
    try:
        # Check if any auth-related endpoints exist
        response = requests.options(f"{BASE_URL}/auth/login")
        print(f"OPTIONS /auth/login: {response.status_code}")
        
        if response.status_code == 405:
            print("✅ /auth/login endpoint exists but doesn't accept OPTIONS")
        elif response.status_code == 404:
            print("❌ /auth/login endpoint not found - auth router may not be included")
        
    except Exception as e:
        print(f"Error checking auth router: {e}")

def test_alternative_login_formats():
    """Test different login data formats"""
    print("\nTesting different login formats...")
    
    login_formats = [
        # Form data (OAuth2 standard)
        {"data": {"username": "user1@example.com", "password": "user123"}},
        # JSON data
        {"json": {"username": "user1@example.com", "password": "user123"}},
        {"json": {"email": "user1@example.com", "password": "user123"}},
        # Form data with different field names
        {"data": {"email": "user1@example.com", "password": "user123"}},
    ]
    
    for i, format_data in enumerate(login_formats):
        try:
            response = requests.post(f"{BASE_URL}/auth/login", **format_data)
            print(f"Format {i+1}: {response.status_code}")
            if response.status_code not in [404, 405]:
                print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"Format {i+1} error: {e}")

def main():
    print("🔍 Debugging Authentication Setup")
    print("=" * 50)
    
    # First check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("Make sure you're running: python main_clean.py")
        return
    
    check_available_endpoints()
    test_auth_endpoints()
    check_auth_router_inclusion()
    test_alternative_login_formats()
    
    print("\n" + "=" * 50)
    print("🔧 Troubleshooting Steps:")
    print("1. Check if auth_router is properly imported and included in main_clean.py")
    print("2. Verify auth/endpoints.py exists and has login endpoint")
    print("3. Check http://127.0.0.1:8000/docs to see all available endpoints")
    print("4. Make sure database is properly initialized with test users")

if __name__ == "__main__":
    main()