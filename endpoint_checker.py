# endpoint_checker.py - List all available endpoints in your FastAPI backend
import importlib.util
import sys
from typing import List, Dict
import inspect

def load_backend_module(file_path: str):
    """Load the backend module from file path"""
    spec = importlib.util.spec_from_file_location("backend", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def extract_fastapi_routes(app):
    """Extract all routes from FastAPI app"""
    routes = []
    
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            # Handle regular routes
            methods = list(route.methods)
            if 'HEAD' in methods:
                methods.remove('HEAD')  # Remove HEAD as it's automatically added
            
            route_info = {
                'path': route.path,
                'methods': methods,
                'name': getattr(route, 'name', 'unnamed'),
                'endpoint': getattr(route, 'endpoint', None)
            }
            routes.append(route_info)
        elif hasattr(route, 'routes'):
            # Handle sub-routers (like APIRouter)
            for subroute in route.routes:
                if hasattr(subroute, 'methods') and hasattr(subroute, 'path'):
                    methods = list(subroute.methods)
                    if 'HEAD' in methods:
                        methods.remove('HEAD')
                    
                    # Combine router prefix with route path
                    full_path = route.path.rstrip('/') + subroute.path
                    
                    route_info = {
                        'path': full_path,
                        'methods': methods,
                        'name': getattr(subroute, 'name', 'unnamed'),
                        'endpoint': getattr(subroute, 'endpoint', None)
                    }
                    routes.append(route_info)
    
    return routes

def check_frontend_endpoints():
    """List the endpoints the frontend is trying to access"""
    frontend_endpoints = [
        "POST /chat/analyze",
        "POST /chat/quick-action", 
        "GET /chat/quick-actions/{portfolio_id}",
        "GET /chat/specialists",
        "GET /conversations/{portfolio_id}",
        "GET /conversations/{conversation_id}/messages"
    ]
    return frontend_endpoints

def main():
    backend_file = "main_clean.py"
    
    print("🔍 BACKEND ENDPOINT CHECKER")
    print("=" * 50)
    
    try:
        # Load the backend module
        print(f"Loading backend from: {backend_file}")
        backend = load_backend_module(backend_file)
        
        # Find the FastAPI app instance
        app = None
        for name in dir(backend):
            obj = getattr(backend, name)
            if hasattr(obj, 'routes') and hasattr(obj, 'add_api_route'):
                app = obj
                print(f"Found FastAPI app: {name}")
                break
        
        if not app:
            print("❌ No FastAPI app found in the backend file!")
            return
        
        # Extract all routes
        routes = extract_fastapi_routes(app)
        
        print(f"\n📋 AVAILABLE ENDPOINTS ({len(routes)} found):")
        print("-" * 50)
        
        for route in sorted(routes, key=lambda x: x['path']):
            methods_str = ", ".join(route['methods'])
            endpoint_name = route['endpoint'].__name__ if route['endpoint'] else 'unknown'
            print(f"  {methods_str:<10} {route['path']:<40} [{endpoint_name}]")
        
        # Check what frontend is looking for
        print(f"\n🎯 FRONTEND REQUIREMENTS:")
        print("-" * 50)
        frontend_endpoints = check_frontend_endpoints()
        
        # Check if each frontend endpoint exists
        backend_paths = {f"{','.join(r['methods'])} {r['path']}" for r in routes}
        
        for fe_endpoint in frontend_endpoints:
            method, path = fe_endpoint.split(' ', 1)
            
            # Check for exact match
            exact_match = any(
                method in route['methods'] and route['path'] == path 
                for route in routes
            )
            
            # Check for pattern match (with path parameters)
            pattern_match = any(
                method in route['methods'] and 
                path.replace('{portfolio_id}', '.*').replace('{conversation_id}', '.*') in route['path']
                for route in routes
            )
            
            status = "✅" if exact_match else ("🔶" if pattern_match else "❌")
            print(f"  {status} {fe_endpoint}")
        
        print(f"\n🔧 MISSING ENDPOINTS:")
        print("-" * 50)
        
        missing = []
        for fe_endpoint in frontend_endpoints:
            method, path = fe_endpoint.split(' ', 1)
            exists = any(
                method in route['methods'] and (
                    route['path'] == path or 
                    # Check if it's a parameterized path
                    ('{' in path and route['path'].replace('{', '').replace('}', '') in path.replace('{', '').replace('}', ''))
                )
                for route in routes
            )
            if not exists:
                missing.append(fe_endpoint)
        
        if missing:
            for endpoint in missing:
                print(f"  ❌ {endpoint}")
            
            print(f"\n💡 SUGGESTED BACKEND ADDITIONS:")
            print("-" * 50)
            print("Add these endpoints to your FastAPI backend:")
            for endpoint in missing:
                method, path = endpoint.split(' ', 1)
                func_name = path.replace('/', '_').replace('{', '').replace('}', '').replace('-', '_').strip('_')
                print(f"""
@app.{method.lower()}("{path}")
async def {func_name}():
    # TODO: Implement this endpoint
    return {{"message": "Not implemented yet"}}""")
        else:
            print("  🎉 All frontend endpoints are available!")
        
        print(f"\n📊 SUMMARY:")
        print("-" * 50)
        print(f"  • Total backend endpoints: {len(routes)}")
        print(f"  • Frontend requirements: {len(frontend_endpoints)}")
        print(f"  • Missing endpoints: {len(missing)}")
        
        if missing:
            print(f"\n⚠️  The 403 Forbidden errors are likely because these {len(missing)} endpoints are missing!")
        else:
            print(f"\n✅ All endpoints exist - the 403 errors might be due to authentication or CORS issues.")
            
    except FileNotFoundError:
        print(f"❌ Backend file '{backend_file}' not found!")
        print("Make sure you're running this script from the same directory as your backend file.")
    except Exception as e:
        print(f"❌ Error analyzing backend: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    main()