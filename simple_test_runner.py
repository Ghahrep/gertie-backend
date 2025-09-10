# simple_test_runner.py - Test current API functionality
"""
Simple test runner for existing API functionality
Tests basic endpoints and enhanced features
"""

import time
import requests
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:8000"

def test_basic_endpoints():
    """Test basic API endpoints"""
    print("Testing Basic API Endpoints")
    print("=" * 40)
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"   Enhanced features: {data.get('enhanced_features', {})}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test status endpoint
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status endpoint working - {data.get('status', 'operational')}")
            print(f"   Auth endpoints: {data.get('endpoints', {}).get('auth_endpoints', 0)}")
            print(f"   Total endpoints: {data.get('endpoints', {}).get('total_endpoints', 0)}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")

def test_authentication():
    """Test authentication endpoints"""
    print("\nTesting Authentication")
    print("=" * 40)
    
    # Test debug users endpoint first (to see available users)
    try:
        response = requests.get(f"{BASE_URL}/auth/debug/users")
        if response.status_code == 200:
            data = response.json()
            available_users = data.get("available_users", [])
            print(f"✅ Found {len(available_users)} users in database")
            for user in available_users[:3]:  # Show first 3 users
                print(f"   - {user.get('email', 'unknown')}")
        else:
            print(f"⚠️  Debug users endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Debug users error: {e}")
    
    # Test login endpoint
    try:
        login_data = {
            "username": "user1@example.com",
            "password": "user123"
        }
        response = requests.post(f"{BASE_URL}/auth/login", data=login_data)
        
        if response.status_code == 200:
            data = response.json()
            if "access_token" in data:
                print("✅ Login working - token received")
                print(f"   User: {data.get('user', {}).get('email', 'unknown')}")
                print(f"   Role: {data.get('user', {}).get('role', 'unknown')}")
                return data["access_token"]
            else:
                print("❌ Login response missing token")
                return None
        else:
            print(f"❌ Login failed: {response.status_code}")
            if response.status_code == 422:
                print(f"   Error: {response.json()}")
            elif response.status_code == 429:
                print("   Rate limited - try again in a moment")
            return None
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_protected_endpoints(token):
    """Test protected endpoints with authentication"""
    if not token:
        print("\nSkipping protected endpoint tests - no token")
        return None
    
    print("\nTesting Protected Endpoints")
    print("=" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test user portfolios endpoint
    try:
        response = requests.get(f"{BASE_URL}/user/portfolios", headers=headers)
        if response.status_code == 200:
            data = response.json()
            portfolios = data.get("portfolios", [])
            portfolio_count = len(portfolios)
            print(f"✅ User portfolios endpoint working - {portfolio_count} portfolios found")
            
            # Show portfolio details
            for portfolio in portfolios[:2]:  # Show first 2 portfolios
                print(f"   Portfolio {portfolio['id']}: {portfolio['name']} (${portfolio.get('total_value', 0):,.2f})")
            
            # Return first portfolio ID for further testing
            if portfolio_count > 0:
                return portfolios[0]["id"]
        else:
            print(f"❌ User portfolios endpoint failed: {response.status_code}")
            if response.status_code == 401:
                print("   Authentication token may be invalid")
            elif response.status_code == 403:
                print("   Access denied - check user permissions")
    except Exception as e:
        print(f"❌ User portfolios endpoint error: {e}")
    
    return None

def test_portfolio_operations(token, portfolio_id):
    """Test portfolio-specific operations"""
    if not token or not portfolio_id:
        print("\nSkipping portfolio operation tests")
        return
    
    print(f"\nTesting Portfolio Operations (ID: {portfolio_id})")
    print("=" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test portfolio summary
    try:
        response = requests.get(f"{BASE_URL}/portfolio/{portfolio_id}/summary", headers=headers)
        if response.status_code == 200:
            data = response.json()
            total_value = data.get("total_value", 0)
            print(f"✅ Portfolio summary working - Total value: ${total_value:,.2f}")
            if "holdings_breakdown" in data:
                print(f"   Holdings breakdown available")
        else:
            print(f"❌ Portfolio summary failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Portfolio summary error: {e}")
    
    # Test holdings endpoint
    try:
        response = requests.get(f"{BASE_URL}/portfolio/{portfolio_id}/holdings", headers=headers)
        if response.status_code == 200:
            data = response.json()
            # Check if it's enhanced paginated response
            if "pagination" in data:
                pagination = data["pagination"]
                holding_count = len(pagination["items"])
                total_items = pagination.get("total_items", 0)
                print(f"✅ Holdings endpoint working (Enhanced/Paginated)")
                print(f"   Current page items: {holding_count}")
                print(f"   Total holdings: {total_items}")
                print(f"   Enhanced features available: {data.get('enhanced_features_available', False)}")
            elif "holdings" in data:
                holding_count = len(data["holdings"])
                print(f"✅ Holdings endpoint working (Basic) - {holding_count} holdings")
            else:
                print("✅ Holdings endpoint responding (unknown format)")
        else:
            print(f"❌ Holdings endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Holdings endpoint error: {e}")

def test_analysis_endpoints(token, portfolio_id):
    """Test analysis endpoints"""
    if not token or not portfolio_id:
        print("\nSkipping analysis tests")
        return
    
    print(f"\nTesting Analysis Endpoints (Portfolio ID: {portfolio_id})")
    print("=" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test risk analysis
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze/risk", 
            json={"portfolio_id": portfolio_id},
            headers=headers
        )
        execution_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✅ Risk analysis working - completed in {execution_time:.3f}s")
                
                # Show some results if available
                analysis = data.get("analysis", {})
                if "portfolio_risk" in analysis:
                    risk = analysis["portfolio_risk"]
                    print(f"   Portfolio Risk: {risk:.4f}")
            else:
                print(f"❌ Risk analysis failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Risk analysis endpoint failed: {response.status_code}")
            if response.status_code == 404:
                print("   Check if portfolio access dependency is working")
    except Exception as e:
        print(f"❌ Risk analysis error: {e}")

def test_enhanced_features(token, portfolio_id):
    """Test enhanced API features (Task 2.5)"""
    if not token or not portfolio_id:
        print("\nSkipping enhanced features tests")
        return
    
    print(f"\nTesting Enhanced API Features (Portfolio ID: {portfolio_id})")
    print("=" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test pagination
    print("Testing Pagination:")
    try:
        response = requests.get(
            f"{BASE_URL}/portfolio/{portfolio_id}/holdings?page=1&limit=3",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if "pagination" in data:
                pagination = data["pagination"]
                print("✅ Pagination implemented and working")
                print(f"   Page: {pagination.get('current_page', 'N/A')}")
                print(f"   Items per page: {pagination.get('items_per_page', 'N/A')}")
                print(f"   Total pages: {pagination.get('total_pages', 'N/A')}")
                print(f"   Has next: {pagination.get('has_next', 'N/A')}")
            else:
                print("❌ Pagination not implemented yet")
        else:
            print(f"❌ Pagination test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Pagination test error: {e}")
    
    # Test filtering
    print("\nTesting Filtering:")
    try:
        response = requests.get(
            f"{BASE_URL}/portfolio/{portfolio_id}/holdings?min_value=100&search=A",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if "filters_applied" in data:
                filters = data["filters_applied"]
                print("✅ Filtering implemented and working")
                print(f"   Applied filters: {filters}")
            else:
                print("❌ Filtering not implemented yet")
        else:
            print(f"❌ Filtering test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Filtering test error: {e}")
    
    # Test validation
    print("\nTesting Validation:")
    try:
        # Test invalid portfolio weights
        response = requests.post(
            f"{BASE_URL}/portfolio/{portfolio_id}/trade-orders",
            json={
                "target_weights": {"AAPL": 0.6, "GOOGL": 0.3},  # Only 90%, should fail
                "min_trade_amount": 100
            },
            headers=headers
        )
        
        if response.status_code == 422:
            print("✅ Validation working - rejected invalid weights")
        elif response.status_code == 400:
            error_detail = response.json().get('detail', '')
            if 'weight' in error_detail.lower():
                print("✅ Business rule validation working")
            else:
                print(f"⚠️  Got 400 but unclear if validation: {error_detail}")
        else:
            print(f"⚠️  Validation test unexpected result: {response.status_code}")
    except Exception as e:
        print(f"❌ Validation test error: {e}")

def test_rate_limiting_and_monitoring(token):
    """Test rate limiting and monitoring features"""
    if not token:
        print("\nSkipping rate limiting tests")
        return
    
    print("\nTesting Rate Limiting & Monitoring")
    print("=" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test rate limiting headers
    try:
        response = requests.get(f"{BASE_URL}/health", headers=headers)
        
        rate_limit_headers = {
            'X-RateLimit-Limit': response.headers.get('X-RateLimit-Limit'),
            'X-RateLimit-Remaining': response.headers.get('X-RateLimit-Remaining'),
            'X-Response-Time': response.headers.get('X-Response-Time')
        }
        
        present_headers = {k: v for k, v in rate_limit_headers.items() if v}
        
        if len(present_headers) >= 2:
            print("✅ Rate limiting headers present")
            for header, value in present_headers.items():
                print(f"   {header}: {value}")
        else:
            print("❌ Rate limiting headers missing")
            
    except Exception as e:
        print(f"❌ Rate limiting header test error: {e}")
    
    # Test rate limit status endpoint
    try:
        response = requests.get(f"{BASE_URL}/user/rate-limit-status", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Rate limit status endpoint working")
            rate_limits = data.get('rate_limits', {})
            for endpoint_type, limits in rate_limits.items():
                print(f"   {endpoint_type}: {limits.get('remaining', 'N/A')}/{limits.get('limit', 'N/A')} remaining")
        else:
            print(f"❌ Rate limit status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Rate limit status test error: {e}")

def test_admin_features(token):
    """Test admin-only features (if user has admin role)"""
    if not token:
        return
    
    print("\nTesting Admin Features")
    print("=" * 40)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test system metrics (admin only)
    try:
        response = requests.get(f"{BASE_URL}/admin/metrics/system", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Admin system metrics working")
            metrics = data.get('system_metrics', {})
            print(f"   Uptime: {metrics.get('uptime_seconds', 'N/A')}s")
            print(f"   CPU: {metrics.get('cpu_percent', 'N/A')}%")
        elif response.status_code in [401, 403]:
            print("⚠️  Admin access required (user doesn't have admin role)")
        else:
            print(f"❌ Admin system metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Admin system metrics error: {e}")

def main():
    """Run all tests"""
    print("🚀 Running Comprehensive API Tests")
    print("=" * 50)
    print(f"Testing API at: {BASE_URL}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Basic endpoint tests
    test_basic_endpoints()
    
    # Authentication tests
    token = test_authentication()
    
    # Protected endpoint tests
    portfolio_id = test_protected_endpoints(token)
    
    # Portfolio operation tests
    test_portfolio_operations(token, portfolio_id)
    
    # Analysis tests
    test_analysis_endpoints(token, portfolio_id)
    
    # Enhanced feature tests (Task 2.5)
    test_enhanced_features(token, portfolio_id)
    
    # Rate limiting and monitoring tests
    test_rate_limiting_and_monitoring(token)
    
    # Admin feature tests
    test_admin_features(token)
    
    print("\n" + "=" * 50)
    print("✅ Comprehensive API Testing Complete!")
    print("\nSummary:")
    print("✓ Basic endpoints (health, status)")
    print("✓ Authentication (login, token validation)")
    print("✓ Protected endpoints (portfolios, holdings)")
    print("✓ Analysis endpoints (risk analysis)")
    print("✓ Enhanced features (pagination, filtering, validation)")
    print("✓ Rate limiting and monitoring")
    print("✓ Admin features (if applicable)")
    print("\nYour Task 2.5: Enhanced API Features implementation is ready for testing!")

if __name__ == "__main__":
    main()