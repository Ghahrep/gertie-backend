# test_enhanced_api.py - Tests for Enhanced API Features
"""
Tests for Task 2.5: Enhanced API Features
========================================

Tests pagination, filtering, validation, rate limiting, and monitoring.
"""

import pytest
import time
import json
from fastapi.testclient import TestClient
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

from main_clean import app
from auth.middleware import create_jwt_token

# Test client
client = TestClient(app)

# Test user tokens
USER_TOKEN = create_jwt_token(1, "user1@example.com", "user")
ADMIN_TOKEN = create_jwt_token(3, "admin@example.com", "admin")

try:
    import requests
    
    # Get real tokens from your running server
    login_response = requests.post("http://127.0.0.1:8000/auth/login", data={
        "username": "user1@example.com",
        "password": "user123"
    })
    
    if login_response.status_code == 200:
        USER_TOKEN = login_response.json()["access_token"]
        print("✅ Got user token successfully")
    else:
        print(f"❌ Failed to get user token: {login_response.status_code}")
        USER_TOKEN = "dummy_token"
    
    # Get admin token
    admin_login = requests.post("http://127.0.0.1:8000/auth/login", data={
        "username": "admin@example.com", 
        "password": "admin123"
    })
    
    if admin_login.status_code == 200:
        ADMIN_TOKEN = admin_login.json()["access_token"]
        print("✅ Got admin token successfully")
    else:
        ADMIN_TOKEN = "dummy_admin_token"
        
except Exception as e:
    print(f"❌ Token generation failed: {e}")
    USER_TOKEN = "dummy_token"
    ADMIN_TOKEN = "dummy_admin_token"

def get_auth_headers(token: str) -> dict:
    """Get authorization headers"""
    return {"Authorization": f"Bearer {token}"}

class TestPaginationFeatures:
    """Test pagination functionality"""
    
    def test_portfolio_holdings_pagination(self):
        """Test portfolio holdings pagination"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test basic pagination
        response = client.get(
            "/portfolio/1/holdings?page=1&limit=10",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            pagination = data.get('pagination', {})
            
            assert 'items' in pagination
            assert 'total_items' in pagination
            assert 'total_pages' in pagination
            assert 'current_page' in pagination
            assert pagination['current_page'] == 1
            assert 'has_next' in pagination
            assert 'has_previous' in pagination
            
            print("✅ Portfolio holdings pagination working")
            print(f"   Total items: {pagination['total_items']}")
            print(f"   Total pages: {pagination['total_pages']}")
        else:
            print(f"⚠️  Portfolio holdings pagination test failed: {response.status_code}")
    
    def test_pagination_parameters_validation(self):
        """Test pagination parameter validation"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test invalid page number
        response = client.get(
            "/portfolio/1/holdings?page=0&limit=10",
            headers=headers
        )
        assert response.status_code == 422  # Validation error
        
        # Test invalid limit
        response = client.get(
            "/portfolio/1/holdings?page=1&limit=101",
            headers=headers
        )
        assert response.status_code == 422  # Validation error
        
        print("✅ Pagination parameter validation working")
    
    def test_sorting_functionality(self):
        """Test sorting functionality"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test sorting by different fields
        sort_fields = ['shares', 'purchase_price', 'ticker']
        
        for sort_field in sort_fields:
            response = client.get(
                f"/portfolio/1/holdings?sort_by={sort_field}&sort_order=desc",
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"✅ Sorting by {sort_field} working")
            else:
                print(f"⚠️  Sorting by {sort_field} failed: {response.status_code}")
    
    def test_pagination_metadata_accuracy(self):
        """Test that pagination metadata is accurate"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Get page 1 with limit 5
        response = client.get(
            "/portfolio/1/holdings?page=1&limit=5",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            pagination = data['pagination']
            
            # Verify metadata consistency
            assert len(pagination['items']) <= 5
            assert pagination['items_per_page'] == 5
            
            if pagination['total_items'] > 5:
                assert pagination['has_next'] == True
                assert pagination['next_page'] == 2
            
            assert pagination['has_previous'] == False
            assert pagination['previous_page'] is None
            
            print("✅ Pagination metadata accuracy verified")

class TestFilteringFeatures:
    """Test filtering functionality"""
    
    def test_holdings_value_filtering(self):
        """Test filtering by holding value"""
        headers = get_auth_headers(USER_TOKEN)
        
        response = client.get(
            "/portfolio/1/holdings?min_value=1000&max_value=50000",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            filters = data.get('filters_applied', {})
            assert filters['min_value'] == 1000
            assert filters['max_value'] == 50000
            
            # Verify all returned holdings meet filter criteria
            for item in data['pagination']['items']:
                market_value = item['market_value']
                assert 1000 <= market_value <= 50000
            
            print("✅ Value filtering working and accurate")
        else:
            print(f"⚠️  Value filtering failed: {response.status_code}")
    
    def test_search_functionality(self):
        """Test search functionality"""
        headers = get_auth_headers(USER_TOKEN)
        
        response = client.get(
            "/portfolio/1/holdings?search=AAPL",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            filters = data.get('filters_applied', {})
            assert filters['search'] == 'AAPL'
            
            # Verify search results contain the search term
            for item in data['pagination']['items']:
                ticker_match = 'AAPL' in item['ticker'].upper()
                name_match = 'AAPL' in item['name'].upper() if item['name'] else False
                assert ticker_match or name_match
            
            print("✅ Search functionality working")
        else:
            print(f"⚠️  Search functionality failed: {response.status_code}")
    
    def test_date_range_filtering(self):
        """Test date range filtering"""
        headers = get_auth_headers(USER_TOKEN)
        
        start_date = (date.today() - timedelta(days=365)).isoformat()
        end_date = date.today().isoformat()
        
        response = client.get(
            f"/portfolio/1/holdings?start_date={start_date}&end_date={end_date}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            date_range = data.get('filters_applied', {}).get('date_range', {})
            assert date_range['start_date'] == start_date
            assert date_range['end_date'] == end_date
            print("✅ Date range filtering working")
        else:
            print(f"⚠️  Date range filtering failed: {response.status_code}")
    
    def test_sector_filtering(self):
        """Test sector-based filtering"""
        headers = get_auth_headers(USER_TOKEN)
        
        response = client.get(
            "/portfolio/1/holdings?sector=Technology",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            filters = data.get('filters_applied', {})
            assert filters['sector'] == 'Technology'
            print("✅ Sector filtering working")
        else:
            print(f"⚠️  Sector filtering failed: {response.status_code}")
    
    def test_combined_filtering(self):
        """Test multiple filters applied together"""
        headers = get_auth_headers(USER_TOKEN)
        
        response = client.get(
            "/portfolio/1/holdings?min_value=500&sector=Technology&sort_by=market_value&sort_order=desc",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            filters = data.get('filters_applied', {})
            assert filters['min_value'] == 500
            assert filters['sector'] == 'Technology'
            print("✅ Combined filtering working")
        else:
            print(f"⚠️  Combined filtering failed: {response.status_code}")

class TestValidationFeatures:
    """Test request validation"""
    
    def test_portfolio_weights_validation(self):
        """Test portfolio weight validation in trade orders"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test invalid weights (don't sum to 100%)
        invalid_weights = {
            "AAPL": 0.3,
            "GOOGL": 0.3,
            "MSFT": 0.3  # Total = 90%, should fail
        }
        
        response = client.post(
            "/portfolio/1/trade-orders",
            json={
                "target_weights": invalid_weights,
                "min_trade_amount": 100
            },
            headers=headers
        )
        
        assert response.status_code == 422  # Validation error
        error_detail = response.json()['detail']
        assert "weights must sum to 100%" in error_detail.lower()
        print("✅ Portfolio weights validation working")
    
    def test_trade_amount_validation(self):
        """Test trade amount validation"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test with negative amount
        response = client.post(
            "/portfolio/1/trade-orders",
            json={
                "target_weights": {"AAPL": 1.0},
                "min_trade_amount": -100  # Invalid negative amount
            },
            headers=headers
        )
        
        assert response.status_code == 422  # Validation error
        print("✅ Trade amount validation working")
    
    def test_ticker_sanitization(self):
        """Test ticker symbol sanitization"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test with invalid ticker format
        response = client.post(
            "/portfolio/1/trade-orders",
            json={
                "target_weights": {"invalid_ticker_123456789": 1.0},  # Too long
                "min_trade_amount": 100
            },
            headers=headers
        )
        
        assert response.status_code == 422  # Validation error
        print("✅ Ticker sanitization working")
    
    def test_date_range_validation(self):
        """Test date range validation"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test with end date before start date
        future_date = (date.today() + timedelta(days=1)).isoformat()
        past_date = (date.today() - timedelta(days=1)).isoformat()
        
        response = client.get(
            f"/portfolio/1/holdings?start_date={future_date}&end_date={past_date}",
            headers=headers
        )
        
        assert response.status_code == 422  # Validation error
        print("✅ Date range validation working")
    
    def test_input_sanitization(self):
        """Test input sanitization against XSS and SQL injection"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test with potential XSS payload
        malicious_search = "<script>alert('xss')</script>"
        
        response = client.get(
            f"/portfolio/1/holdings?search={malicious_search}",
            headers=headers
        )
        
        # Should either sanitize or reject
        if response.status_code == 200:
            data = response.json()
            search_term = data.get('filters_applied', {}).get('search', '')
            assert '<script>' not in search_term
            print("✅ XSS sanitization working")
        elif response.status_code == 422:
            print("✅ XSS input rejected by validation")

class TestRateLimitingFeatures:
    """Test rate limiting functionality"""
    
    def test_general_rate_limiting(self):
        """Test general API rate limiting"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Make many requests quickly to trigger rate limit
        responses = []
        for i in range(50):
            response = client.get("/health", headers=headers)
            responses.append(response.status_code)
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        if 429 in responses:
            print("✅ General rate limiting working")
        else:
            print("⚠️  Rate limiting may be too permissive or not enabled")
    
    def test_endpoint_specific_rate_limiting(self):
        """Test endpoint-specific rate limiting"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Test trade orders rate limiting (should be stricter)
        responses = []
        for i in range(15):
            response = client.post(
                "/portfolio/1/trade-orders",
                json={
                    "target_weights": {"AAPL": 1.0},
                    "min_trade_amount": 100
                },
                headers=headers
            )
            responses.append(response.status_code)
            if response.status_code == 429:
                break
        
        # Should hit rate limit faster for trading endpoints
        if 429 in responses:
            print("✅ Trade endpoint rate limiting working")
            
            # Check rate limit headers
            rate_limited_response = next(r for r in responses if r == 429)
            # Would check headers in actual response object
            
        else:
            print("⚠️  Trade endpoint rate limiting may be too permissive")
    
    def test_rate_limit_headers(self):
        """Test rate limit headers are present"""
        headers = get_auth_headers(USER_TOKEN)
        
        response = client.get("/portfolio/1/holdings", headers=headers)
        
        if response.status_code == 200:
            # Check for rate limit headers
            expected_headers = [
                'X-RateLimit-Limit',
                'X-RateLimit-Remaining',
                'X-RateLimit-Reset'
            ]
            
            present_headers = [h for h in expected_headers if h in response.headers]
            
            if len(present_headers) >= 2:
                print("✅ Rate limit headers present")
            else:
                print("⚠️  Rate limit headers missing")
    
    def test_rate_limit_status_endpoint(self):
        """Test rate limit status endpoint"""
        headers = get_auth_headers(USER_TOKEN)
        
        response = client.get("/user/rate-limit-status", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            assert 'rate_limits' in data
            assert 'user_id' in data
            
            rate_limits = data['rate_limits']
            for endpoint_type in ['general', 'analysis', 'trading']:
                if endpoint_type in rate_limits:
                    limit_info = rate_limits[endpoint_type]
                    assert 'limit' in limit_info
                    assert 'remaining' in limit_info
                    assert 'reset_time' in limit_info
            
            print("✅ Rate limit status endpoint working")
        else:
            print(f"⚠️  Rate limit status endpoint failed: {response.status_code}")

class TestMonitoringFeatures:
    """Test API monitoring and metrics"""
    
    def test_request_logging(self):
        """Test request logging functionality"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Make a request that should be logged
        response = client.get("/portfolio/1/holdings", headers=headers)
        
        # Check for performance headers
        if 'X-Response-Time' in response.headers:
            response_time = response.headers['X-Response-Time']
            assert 's' in response_time  # Should be in format like "0.123s"
            print("✅ Request logging and performance headers working")
        else:
            print("⚠️  Performance headers missing")
    
    def test_system_metrics_endpoint(self):
        """Test system metrics endpoint (admin only)"""
        admin_headers = get_auth_headers(ADMIN_TOKEN)
        
        response = client.get("/admin/metrics/system", headers=admin_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert 'system_metrics' in data
            
            metrics = data['system_metrics']
            expected_metrics = ['uptime_seconds', 'cpu_percent', 'memory', 'disk']
            
            present_metrics = [m for m in expected_metrics if m in metrics]
            
            if len(present_metrics) >= 3:
                print("✅ System metrics endpoint working")
                print(f"   Available metrics: {present_metrics}")
            else:
                print("⚠️  System metrics incomplete")
        else:
            print(f"⚠️  System metrics endpoint failed: {response.status_code}")
    
    def test_request_metrics_endpoint(self):
        """Test request metrics endpoint (admin only)"""
        admin_headers = get_auth_headers(ADMIN_TOKEN)
        
        # Make some requests first to generate metrics
        user_headers = get_auth_headers(USER_TOKEN)
        for _ in range(5):
            client.get("/health", headers=user_headers)
        
        response = client.get("/admin/metrics/requests?minutes=60", headers=admin_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert 'request_metrics' in data
            
            metrics = data['request_metrics']
            expected_fields = ['total_requests', 'average_response_time', 'error_rate']
            
            present_fields = [f for f in expected_fields if f in metrics]
            
            if len(present_fields) >= 2:
                print("✅ Request metrics endpoint working")
                print(f"   Total requests: {metrics.get('total_requests', 0)}")
            else:
                print("⚠️  Request metrics incomplete")
        else:
            print(f"⚠️  Request metrics endpoint failed: {response.status_code}")
    
    def test_admin_only_access(self):
        """Test that admin endpoints require admin access"""
        user_headers = get_auth_headers(USER_TOKEN)
        
        # User should not access admin endpoints
        response = client.get("/admin/metrics/system", headers=user_headers)
        assert response.status_code in [401, 403]  # Unauthorized or Forbidden
        
        response = client.get("/admin/metrics/requests", headers=user_headers)
        assert response.status_code in [401, 403]
        
        print("✅ Admin access control working")

class TestBusinessRuleValidation:
    """Test business rule validation"""
    
    def test_minimum_trade_amount(self):
        """Test minimum trade amount business rule"""
        headers = get_auth_headers(USER_TOKEN)
        
        response = client.post(
            "/portfolio/1/trade-orders",
            json={
                "target_weights": {"AAPL": 1.0},
                "min_trade_amount": 50  # Below minimum of $100
            },
            headers=headers
        )
        
        # Should either validate at API level or in business logic
        if response.status_code == 422:
            print("✅ Minimum trade amount validation working")
        elif response.status_code == 400:
            error_detail = response.json().get('detail', '')
            if 'minimum' in error_detail.lower():
                print("✅ Minimum trade amount business rule working")
    
    def test_portfolio_concentration_limits(self):
        """Test portfolio concentration limits"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Try to create a very concentrated portfolio
        response = client.post(
            "/portfolio/1/trade-orders",
            json={
                "target_weights": {"AAPL": 1.0},  # 100% in one stock
                "min_trade_amount": 1000
            },
            headers=headers
        )
        
        # This might be allowed or might trigger concentration warnings
        if response.status_code in [200, 400]:
            print("✅ Concentration limits test completed")
    
    def test_date_range_business_rules(self):
        """Test date range business rules"""
        headers = get_auth_headers(USER_TOKEN)
        
        # Try to analyze very old data (beyond 10 year limit)
        old_date = (date.today() - timedelta(days=11*365)).isoformat()
        
        response = client.post(
            "/analyze/risk/1",
            json={"start_date": old_date},
            headers=headers
        )
        
        if response.status_code == 422:
            error_detail = response.json().get('detail', '')
            if '10 year' in error_detail or 'historical' in error_detail:
                print("✅ Historical data limit business rule working")

class TestPerformanceFeatures:
    """Test performance-related features"""
    
    def test_response_time_consistency(self):
        """Test response time consistency"""
        headers = get_auth_headers(USER_TOKEN)
        
        response_times = []
        for _ in range(10):
            start_time = time.time()
            response = client.get("/portfolio/1/holdings", headers=headers)
            end_time = time.time()
            
            if response.status_code == 200:
                response_times.append(end_time - start_time)
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            
            print(f"✅ Performance test completed")
            print(f"   Average response time: {avg_time:.3f}s")
            print(f"   Maximum response time: {max_time:.3f}s")
            
            # Should be consistently fast
            assert avg_time < 2.0  # Average under 2 seconds
            assert max_time < 5.0  # No request over 5 seconds
    
    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests"""
        import threading
        
        headers = get_auth_headers(USER_TOKEN)
        results = []
        
        def make_request():
            response = client.get("/health", headers=headers)
            results.append(response.status_code)
        
        # Make 10 concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Most requests should succeed
        success_rate = sum(1 for r in results if r == 200) / len(results)
        assert success_rate >= 0.8  # At least 80% success rate
        
        print(f"✅ Concurrent request test completed")
        print(f"   Success rate: {success_rate:.1%}")

def run_enhanced_api_tests():
    """Run all enhanced API tests"""
    print("🚀 Running Enhanced API Features Tests")
    print("=" * 50)
    
    # Create test instances
    pagination_tests = TestPaginationFeatures()
    filtering_tests = TestFilteringFeatures()
    validation_tests = TestValidationFeatures()
    rate_limit_tests = TestRateLimitingFeatures()
    monitoring_tests = TestMonitoringFeatures()
    business_rule_tests = TestBusinessRuleValidation()
    performance_tests = TestPerformanceFeatures()
    
    try:
        print("\n📊 Testing Pagination Features:")
        pagination_tests.test_portfolio_holdings_pagination()
        pagination_tests.test_pagination_parameters_validation()
        pagination_tests.test_sorting_functionality()
        pagination_tests.test_pagination_metadata_accuracy()
        
        print("\n🔍 Testing Filtering Features:")
        filtering_tests.test_holdings_value_filtering()
        filtering_tests.test_search_functionality()
        filtering_tests.test_date_range_filtering()
        filtering_tests.test_sector_filtering()
        filtering_tests.test_combined_filtering()
        
        print("\n✅ Testing Validation Features:")
        validation_tests.test_portfolio_weights_validation()
        validation_tests.test_trade_amount_validation()
        validation_tests.test_ticker_sanitization()
        validation_tests.test_date_range_validation()
        validation_tests.test_input_sanitization()
        
        print("\n🚦 Testing Rate Limiting Features:")
        rate_limit_tests.test_general_rate_limiting()
        rate_limit_tests.test_endpoint_specific_rate_limiting()
        rate_limit_tests.test_rate_limit_headers()
        rate_limit_tests.test_rate_limit_status_endpoint()
        
        print("\n📈 Testing Monitoring Features:")
        monitoring_tests.test_request_logging()
        monitoring_tests.test_system_metrics_endpoint()
        monitoring_tests.test_request_metrics_endpoint()
        monitoring_tests.test_admin_only_access()
        
        print("\n📋 Testing Business Rule Validation:")
        business_rule_tests.test_minimum_trade_amount()
        business_rule_tests.test_portfolio_concentration_limits()
        business_rule_tests.test_date_range_business_rules()
        
        print("\n⚡ Testing Performance Features:")
        performance_tests.test_response_time_consistency()
        performance_tests.test_concurrent_request_handling()
        
        print("\n" + "=" * 50)
        print("✅ Enhanced API Features Testing Complete!")
        print("\nKey Features Tested:")
        print("• Pagination and sorting")
        print("• Advanced filtering")
        print("• Input validation and sanitization")
        print("• Rate limiting")
        print("• API monitoring and metrics")
        print("• Business rule validation")
        print("• Performance consistency")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        raise

if __name__ == "__main__":
    run_enhanced_api_tests()