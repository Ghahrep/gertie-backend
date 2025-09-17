# simple_endpoint_checker.py - Simple endpoint checker for FastAPI backend
import requests
import json
from typing import List, Dict

def test_endpoint(base_url: str, method: str, path: str, test_data=None):
    """Test if an endpoint exists and what it returns"""
    url = f"{base_url}{path}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=test_data or {}, timeout=5)
        elif method == "DELETE":
            response = requests.delete(url, timeout=5)
        else:
            return {"status": "unknown_method", "code": None}
        
        return {
            "status": "exists" if response.status_code != 404 else "not_found",
            "code": response.status_code,
            "response_size": len(response.content)
        }
    except requests.exceptions.ConnectionError:
        return {"status": "connection_error", "code": None}
    except requests.exceptions.Timeout:
        return {"status": "timeout", "code": None}
    except Exception as e:
        return {"status": "error", "code": None, "error": str(e)}

def check_frontend_requirements():
    """Check the specific endpoints the frontend needs"""
    base_url = "http://localhost:8000"
    
    # Test portfolio ID - using 3 based on your console logs
    portfolio_id = "3"
    conversation_id = "1"
    
    endpoints_to_test = [
        ("POST", "/chat/analyze", {"query": "test", "portfolio_context": {}}),
        ("POST", "/chat/quick-action", {"action_id": "test", "portfolio_id": portfolio_id}),
        ("GET", f"/chat/quick-actions/{portfolio_id}", None),
        ("GET", "/chat/specialists", None),
        ("GET", f"/conversations/{portfolio_id}", None),
        ("GET", f"/conversations/{conversation_id}/messages", None),
        ("GET", f"/chat/conversations/{portfolio_id}", None),
        ("POST", f"/chat/quick-actions/portfolio_review/execute", None),
    ]
    
    print("🔍 TESTING BACKEND ENDPOINTS")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"Test Portfolio ID: {portfolio_id}")
    print()
    
    results = []
    
    for method, path, test_data in endpoints_to_test:
        print(f"Testing: {method} {path}")
        result = test_endpoint(base_url, method, path, test_data)
        results.append((method, path, result))
        
        status_emoji = {
            "exists": "✅",
            "not_found": "❌", 
            "connection_error": "🔌",
            "timeout": "⏰",
            "error": "💥"
        }.get(result["status"], "❓")
        
        status_code = f" (HTTP {result['code']})" if result["code"] else ""
        print(f"  {status_emoji} {result['status']}{status_code}")
        
        if result["status"] == "exists" and result["code"] == 403:
            print(f"    ⚠️  Endpoint exists but returns 403 Forbidden")
        elif result["status"] == "exists" and result["code"] == 422:
            print(f"    ℹ️  Endpoint exists but needs different parameters")
        elif result["status"] == "exists" and result["code"] in [200, 201]:
            print(f"    🎉 Working correctly!")
        
        print()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 60)
    
    working = sum(1 for _, _, r in results if r["status"] == "exists" and r["code"] in [200, 201, 422])
    forbidden = sum(1 for _, _, r in results if r["status"] == "exists" and r["code"] == 403)
    not_found = sum(1 for _, _, r in results if r["status"] == "not_found")
    errors = sum(1 for _, _, r in results if r["status"] in ["connection_error", "timeout", "error"])
    
    print(f"✅ Working endpoints: {working}")
    print(f"🔒 Forbidden (403): {forbidden}")
    print(f"❌ Not found (404): {not_found}")
    print(f"💥 Connection/Other errors: {errors}")
    
    if forbidden > 0:
        print(f"\n⚠️  DIAGNOSIS: You have {forbidden} endpoints returning 403 Forbidden")
        print("This usually means:")
        print("  1. Authentication is required but not provided")
        print("  2. The endpoint exists but has authorization middleware")
        print("  3. CORS is blocking the request")
        
    if not_found > 0:
        print(f"\n❌ DIAGNOSIS: You have {not_found} endpoints that don't exist")
        print("These endpoints need to be added to your backend:")
        for method, path, result in results:
            if result["status"] == "not_found":
                print(f"  • {method} {path}")
    
    if errors > 0:
        print(f"\n🔌 DIAGNOSIS: Connection issues detected")
        print("Make sure your backend is running on http://localhost:8000")
        print("Check with: python clean_main.py")
    
    # Specific fixes
    print(f"\n💡 QUICK FIXES")
    print("=" * 60)
    
    # Check if the correct endpoints exist vs what frontend is calling
    frontend_calls = [
        "/chat/quick-actions/{portfolio_id}",
        "/chat/specialists", 
        "/conversations/{portfolio_id}"
    ]
    
    backend_paths = [
        "/chat/quick-actions/{portfolio_id}",
        "/chat/specialists",
        "/chat/conversations/{portfolio_id}"  # Note the difference!
    ]
    
    print("Frontend is calling:")
    for path in frontend_calls:
        print(f"  • GET {path}")
    
    print(f"\nBut your backend has:")
    for path in backend_paths:
        print(f"  • GET {path}")
    
    print(f"\n🔧 The issue is endpoint mismatch!")
    print("Your frontend calls: GET /conversations/{portfolio_id}")
    print("Your backend has:    GET /chat/conversations/{portfolio_id}")
    print(f"\nFix: Update your frontend service to use the correct paths!")

if __name__ == "__main__":
    check_frontend_requirements()