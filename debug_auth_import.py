# debug_auth_import.py - Debug authentication router import
"""
Check if auth router is being imported and included correctly
"""

print("🔍 Debugging Authentication Router Import")
print("=" * 50)

# Test 1: Check if auth_router can be imported
print("1. Testing auth_router import...")
try:
    from auth.endpoints import auth_router
    print("✅ auth_router imported successfully")
    print(f"   Router type: {type(auth_router)}")
    print(f"   Router prefix: {getattr(auth_router, 'prefix', 'No prefix')}")
    
    # Check routes
    routes = getattr(auth_router, 'routes', [])
    print(f"   Number of routes: {len(routes)}")
    for route in routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"   Route: {route.methods} {route.path}")
        
except Exception as e:
    print(f"❌ auth_router import failed: {e}")
    print("   This is the problem - fix auth/endpoints.py first")

# Test 2: Check auth middleware imports
print("\n2. Testing auth middleware imports...")
try:
    from auth.middleware import (
        get_current_active_user,
        get_user_portfolio,
        get_user_portfolio_write,
        check_portfolio_access,
        get_admin_user
    )
    print("✅ Auth middleware functions imported successfully")
except Exception as e:
    print(f"❌ Auth middleware import failed: {e}")

# Test 3: Check if main_clean.py includes the router properly
print("\n3. Testing main_clean.py router inclusion...")
try:
    # Read main_clean.py to check router inclusion
    with open('main_clean.py', 'r') as f:
        content = f.read()
    
    if 'app.include_router(auth_router)' in content:
        print("✅ Found app.include_router(auth_router) in main_clean.py")
    else:
        print("❌ app.include_router(auth_router) not found in main_clean.py")
    
    if 'from auth.endpoints import auth_router' in content:
        print("✅ Found auth_router import in main_clean.py")
    else:
        print("❌ auth_router import not found in main_clean.py")
        
except Exception as e:
    print(f"❌ Error reading main_clean.py: {e}")

# Test 4: Try to manually create the auth app to see what happens
print("\n4. Testing manual FastAPI app with auth router...")
try:
    from fastapi import FastAPI
    from auth.endpoints import auth_router
    
    test_app = FastAPI()
    test_app.include_router(auth_router)
    
    # Get routes
    routes = []
    for route in test_app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append(f"{list(route.methods)} {route.path}")
    
    print("✅ Manual app creation successful")
    print("   Routes in test app:")
    for route in routes:
        print(f"   {route}")
        
except Exception as e:
    print(f"❌ Manual app creation failed: {e}")
    print(f"   Error details: {str(e)}")

print("\n" + "=" * 50)
print("🔧 Next Steps:")
print("1. If auth_router import fails, check auth/endpoints.py syntax")
print("2. If import works but routes missing, check router creation")
print("3. If everything works in test, check main_clean.py execution order")
print("4. Restart the server after any fixes")