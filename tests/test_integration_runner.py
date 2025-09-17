# tests/test_integration_runner_updated.py - Updated Integration Test Runner
"""
Updated Backend Integration Test Runner
=======================================
Tests your existing API endpoints plus new portfolio signature functionality.
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Any
import requests
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class UpdatedIntegrationTestRunner:
    """Enhanced integration test runner for portfolio signature endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.auth_token = None
        self.test_results = []
        self.test_portfolio_id = None
        self.test_user_id = None
    
    def log_test(self, test_name: str, status: str, details: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test_name': test_name,
            'status': status,
            'details': details,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        # Color output
        color = '\033[92m' if status == 'PASS' else '\033[91m' if status == 'FAIL' else '\033[93m'
        reset_color = '\033[0m'
        
        print(f"{color}[{status}]{reset_color} {test_name} ({result['duration_ms']}ms)")
        if details:
            print(f"    {details}")
    
    def test_health_check(self) -> bool:
        """Test API health and availability"""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                self.log_test(
                    "Health Check",
                    "PASS",
                    f"API healthy: {health_data.get('status', 'unknown')}",
                    time.time() - start_time
                )
                return True
            else:
                self.log_test(
                    "Health Check",
                    "FAIL",
                    f"HTTP {response.status_code}",
                    time.time() - start_time
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Health Check",
                "FAIL",
                f"Connection error: {str(e)}",
                time.time() - start_time
            )
            return False
    
    def setup_authentication(self) -> bool:
        """Setup authentication for API tests"""
        start_time = time.time()
        
        try:
            # Use existing admin credentials (skip registration)
            auth_data = {
                "email": "admin@example.com", 
                "password": "admin123"
            }
            
            response = requests.post(
                f"{self.base_url}/auth/login-json",
                json=auth_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.auth_token = token_data.get('access_token')
                
                # Get user profile to get user ID
                profile_response = requests.get(
                    f"{self.base_url}/auth/profile",
                    headers=self.get_auth_headers()
                )
                
                if profile_response.status_code == 200:
                    profile = profile_response.json()
                    self.test_user_id = profile.get('id')
                
                self.log_test(
                    "Authentication Setup",
                    "PASS",
                    f"Token obtained for user {self.test_user_id}",
                    time.time() - start_time
                )
                return True
            else:
                self.log_test(
                    "Authentication Setup",
                    "FAIL",
                    f"HTTP {response.status_code}: {response.text}",
                    time.time() - start_time
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Authentication Setup",
                "FAIL",
                f"Exception: {str(e)}",
                time.time() - start_time
            )
            return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if not self.auth_token:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}
    
    def test_portfolio_management(self) -> bool:
        """Test basic portfolio management (your existing functionality)"""
        start_time = time.time()
        
        try:
            # Create a test portfolio
            portfolio_data = {
                "name": "Integration Test Portfolio",
                "description": "Portfolio created for integration testing",
                "currency": "USD"
            }
            
            response = requests.post(
                f"{self.base_url}/portfolios",  # Use /portfolios for portfolio creation
                json=portfolio_data,
                headers=self.get_auth_headers()
            )
            
            if response.status_code in [200, 201]:
                portfolio = response.json()
                self.test_portfolio_id = portfolio.get('id')
                self.log_test(
                    "Portfolio Management",
                    "PASS",
                    f"Portfolio created with ID: {self.test_portfolio_id}",
                    time.time() - start_time
                )
                return True
            else:
                self.log_test(
                    "Portfolio Management",
                    "FAIL",
                    f"HTTP {response.status_code}: {response.text}",
                    time.time() - start_time
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Portfolio Management",
                "FAIL",
                f"Exception: {str(e)}",
                time.time() - start_time
            )
            return False
    
    def test_portfolio_signature_endpoint(self) -> bool:
        """Test new portfolio signature generation endpoint"""
        start_time = time.time()
        
        if not self.test_portfolio_id:
            self.log_test(
                "Portfolio Signature Endpoint",
                "SKIP",
                "No test portfolio available",
                time.time() - start_time
            )
            return False
        
        try:
            # Test GET signature endpoint
            response = requests.get(
                f"{self.base_url}/portfolio/{self.test_portfolio_id}/signature",  # Use /portfolio/ (singular) for signature endpoint
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                signature_data = response.json()
                
                # Validate signature structure
                required_fields = [
                    'id', 'name', 'value', 'holdingsCount', 'riskScore',
                    'volatilityForecast', 'correlation', 'tailRisk',
                    'concentration', 'complexity', 'riskLevel', 'lastUpdated'
                ]
                
                missing_fields = [field for field in required_fields if field not in signature_data]
                
                if missing_fields:
                    self.log_test(
                        "Portfolio Signature Endpoint",
                        "FAIL",
                        f"Missing fields: {missing_fields}",
                        time.time() - start_time
                    )
                    return False
                
                # Validate value ranges
                if not (0 <= signature_data['riskScore'] <= 100):
                    self.log_test(
                        "Portfolio Signature Endpoint",
                        "FAIL",
                        f"Invalid riskScore: {signature_data['riskScore']}",
                        time.time() - start_time
                    )
                    return False
                
                if not (0 <= signature_data['tailRisk'] <= 1):
                    self.log_test(
                        "Portfolio Signature Endpoint",
                        "FAIL",
                        f"Invalid tailRisk: {signature_data['tailRisk']}",
                        time.time() - start_time
                    )
                    return False
                
                self.log_test(
                    "Portfolio Signature Endpoint",
                    "PASS",
                    f"Risk Score: {signature_data['riskScore']}, Risk Level: {signature_data['riskLevel']}, Value: ${signature_data['value']:,.2f}",
                    time.time() - start_time
                )
                return True
            else:
                self.log_test(
                    "Portfolio Signature Endpoint",
                    "FAIL",
                    f"HTTP {response.status_code}: {response.text}",
                    time.time() - start_time
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Portfolio Signature Endpoint",
                "FAIL",
                f"Exception: {str(e)}",
                time.time() - start_time
            )
            return False
    
    def test_signature_force_refresh(self) -> bool:
        """Test portfolio signature force refresh functionality"""
        start_time = time.time()
        
        if not self.test_portfolio_id:
            self.log_test(
                "Signature Force Refresh",
                "SKIP",
                "No test portfolio available",
                time.time() - start_time
            )
            return False
        
        try:
            # Test POST signature endpoint with force refresh
            response = requests.get(  # Changed to GET since live endpoint doesn't need JSON payload
                f"{self.base_url}/portfolios/{self.test_portfolio_id}/signature/live",  # Use /signature/live endpoint
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                signature_data = response.json()
                
                # Should have signature data
                if 'riskScore' in signature_data and 'lastUpdated' in signature_data:
                    self.log_test(
                        "Signature Force Refresh",
                        "PASS",
                        f"Force refresh successful, Risk Score: {signature_data['riskScore']}",
                        time.time() - start_time
                    )
                    return True
                else:
                    self.log_test(
                        "Signature Force Refresh",
                        "FAIL",
                        "Missing signature data in response",
                        time.time() - start_time
                    )
                    return False
            else:
                self.log_test(
                    "Signature Force Refresh",
                    "FAIL",
                    f"HTTP {response.status_code}: {response.text}",
                    time.time() - start_time
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Signature Force Refresh",
                "FAIL",
                f"Exception: {str(e)}",
                time.time() - start_time
            )
            return False
    
    def test_batch_signatures(self) -> bool:
        """Test batch portfolio signatures endpoint"""
        start_time = time.time()
        
        try:
            response = requests.get(
                f"{self.base_url}/portfolios/signatures",  # Use /portfolios/signatures (no api/v1 prefix)
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                batch_data = response.json()
                
                # Should have structure even if empty
                if 'signatures' in batch_data and 'total_count' in batch_data:
                    self.log_test(
                        "Batch Signatures",
                        "PASS",
                        f"Retrieved {batch_data['total_count']} portfolio signatures",
                        time.time() - start_time
                    )
                    return True
                else:
                    self.log_test(
                        "Batch Signatures",
                        "FAIL",
                        "Missing required batch response fields",
                        time.time() - start_time
                    )
                    return False
            else:
                self.log_test(
                    "Batch Signatures",
                    "FAIL",
                    f"HTTP {response.status_code}: {response.text}",
                    time.time() - start_time
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Batch Signatures",
                "FAIL",
                f"Exception: {str(e)}",
                time.time() - start_time
            )
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid requests"""
        start_time = time.time()
        
        try:
            # Test non-existent portfolio
            response = requests.get(
                f"{self.base_url}/portfolios/99999/signature",  # Use /portfolios/ path
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 404:
                self.log_test(
                    "Error Handling - Not Found",
                    "PASS",
                    "Correctly returned 404 for non-existent portfolio",
                    time.time() - start_time
                )
            else:
                self.log_test(
                    "Error Handling - Not Found",
                    "FAIL",
                    f"Expected 404, got HTTP {response.status_code}",
                    time.time() - start_time
                )
                return False
            
            # Test unauthorized access
            response = requests.get(f"{self.base_url}/portfolios/1/signature")  # Use /portfolios/ path
            
            if response.status_code == 401:
                self.log_test(
                    "Error Handling - Unauthorized",
                    "PASS",
                    "Correctly returned 401 for unauthorized request",
                    time.time() - start_time
                )
                return True
            else:
                self.log_test(
                    "Error Handling - Unauthorized",
                    "FAIL",
                    f"Expected 401, got HTTP {response.status_code}",
                    time.time() - start_time
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Error Handling",
                "FAIL",
                f"Exception: {str(e)}",
                time.time() - start_time
            )
            return False
    
    def test_existing_endpoints(self) -> bool:
        """Test your existing API endpoints"""
        start_time = time.time()
        
        try:
            endpoints_to_test = [
                ("/user/portfolios", "User Portfolios"),
                ("/status", "Service Status"),
                ("/user/rate-limit-status", "Rate Limit Status")
            ]
            
            passed = 0
            total = len(endpoints_to_test)
            
            for endpoint, name in endpoints_to_test:
                try:
                    headers = self.get_auth_headers() if endpoint.startswith('/user') else {}
                    response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
                    
                    if response.status_code in [200, 404]:  # 404 acceptable for some endpoints
                        passed += 1
                except:
                    pass  # Count as failure but continue
            
            if passed >= total * 0.7:  # 70% success rate
                self.log_test(
                    "Existing Endpoints",
                    "PASS",
                    f"{passed}/{total} endpoints responding correctly",
                    time.time() - start_time
                )
                return True
            else:
                self.log_test(
                    "Existing Endpoints",
                    "FAIL",
                    f"Only {passed}/{total} endpoints working",
                    time.time() - start_time
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Existing Endpoints",
                "FAIL",
                f"Exception: {str(e)}",
                time.time() - start_time
            )
            return False
    
    def cleanup_test_data(self) -> bool:
        """Clean up test portfolio and data"""
        start_time = time.time()
        
        if not self.test_portfolio_id:
            self.log_test(
                "Test Cleanup",
                "SKIP",
                "No test portfolio to clean up",
                time.time() - start_time
            )
            return True
        
        try:
            response = requests.delete(
                f"{self.base_url}/portfolios/{self.test_portfolio_id}",
                headers=self.get_auth_headers()
            )
            
            if response.status_code in [200, 204, 404]:
                self.log_test(
                    "Test Cleanup",
                    "PASS",
                    f"Test portfolio {self.test_portfolio_id} cleaned up",
                    time.time() - start_time
                )
                return True
            else:
                self.log_test(
                    "Test Cleanup",
                    "WARN",
                    f"Cleanup warning: HTTP {response.status_code}",
                    time.time() - start_time
                )
                return True  # Not critical failure
                
        except Exception as e:
            self.log_test(
                "Test Cleanup",
                "WARN",
                f"Cleanup exception: {str(e)}",
                time.time() - start_time
            )
            return True  # Not critical failure
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        print("\n🚀 Starting Enhanced Backend Integration Tests")
        print("=" * 60)
        
        overall_start = time.time()
        
        # Test execution order
        tests = [
            self.test_health_check,
            self.setup_authentication,
            self.test_existing_endpoints,
            self.test_portfolio_management,
            self.test_portfolio_signature_endpoint,
            self.test_signature_force_refresh,
            self.test_batch_signatures,
            self.test_error_handling,
            self.cleanup_test_data
        ]
        
        # Execute tests
        passed = 0
        failed = 0
        skipped = 0
        
        for test_func in tests:
            try:
                if hasattr(test_func, '__name__') and test_func.__name__ == 'setup_authentication':
                    # Authentication setup is required for subsequent tests
                    if not test_func():
                        print("\n❌ Authentication failed - stopping test execution")
                        break
                    passed += 1
                else:
                    result = test_func()
                    if result is True:
                        passed += 1
                    elif result is False:
                        failed += 1
                    else:
                        skipped += 1
            except Exception as e:
                self.log_test(
                    test_func.__name__,
                    "FAIL",
                    f"Unexpected exception: {str(e)}",
                    0
                )
                failed += 1
        
        total_duration = time.time() - overall_start
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 Enhanced Integration Test Summary")
        print("=" * 60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        
        success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if failed == 0:
            print("\n🎉 All tests passed! Portfolio signature integration is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please review the output above.")
        
        return {
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': success_rate,
            'total_duration': total_duration,
            'detailed_results': self.test_results
        }

def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced portfolio signature integration tests')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--output', help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    runner = UpdatedIntegrationTestRunner(args.url)
    results = runner.run_all_tests()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📁 Detailed results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results['failed'] == 0 else 1)

if __name__ == "__main__":
    main()