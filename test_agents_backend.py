# test_backend.py
import requests
import json
import os
from pprint import pprint
from typing import Dict, Any, List

# --- Configuration ---
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin123"

# --- ANSI Color Codes for better terminal output ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def get_auth_token(session: requests.Session, base_url: str, email: str, password: str) -> str | None:
    """Logs in to the API and returns a JWT token."""
    print(f"{Colors.BLUE}--- Attempting to Authenticate ---{Colors.ENDC}")
    login_url = f"{base_url}/auth/login-json"
    credentials = {"email": email, "password": password}
    try:
        response = session.post(login_url, json=credentials)
        if response.status_code == 200:
            token = response.json().get("access_token")
            print(f"{Colors.GREEN}✅ Authentication successful.{Colors.ENDC}")
            return token
        else:
            print(f"{Colors.RED}❌ Authentication failed. Status: {response.status_code}, Response: {response.text}{Colors.ENDC}")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"{Colors.RED}❌ Connection Error: Could not connect to the API at {login_url}. Is the backend running?{Colors.ENDC}")
        return None

def run_tests(test_cases: List[Dict[str, Any]], base_url: str, auth_token: str | None):
    """Runs a series of API tests."""
    passed_count = 0
    failed_count = 0

    with requests.Session() as session:
        if auth_token:
            session.headers.update({"Authorization": f"Bearer {auth_token}"})
        
        session.headers.update({"Content-Type": "application/json"})

        for i, test in enumerate(test_cases):
            name = test["name"]
            method = test["method"]
            endpoint = test["endpoint"]
            payload = test.get("payload")
            expected_status = test["expected_status"]
            requires_auth = test.get("requires_auth", True)
            
            print(f"\n{Colors.BLUE}--- Test {i+1}/{len(test_cases)}: {name} ---{Colors.ENDC}")

            url = f"{base_url}{endpoint}"
            current_headers = session.headers.copy()

            if not requires_auth and 'Authorization' in current_headers:
                del current_headers['Authorization']

            try:
                response = session.request(
                    method,
                    url,
                    json=payload if method in ["POST", "PUT"] else None,
                    headers=current_headers
                )

                if response.status_code == expected_status:
                    print(f"{Colors.GREEN}✅ PASSED{Colors.ENDC} (Status: {response.status_code})")
                    passed_count += 1
                else:
                    print(f"{Colors.RED}❌ FAILED{Colors.ENDC} (Expected: {expected_status}, Got: {response.status_code})")
                    failed_count += 1
                
                print(f"Response Body:")
                try:
                    pprint(response.json())
                except json.JSONDecodeError:
                    print(response.text)

            except requests.exceptions.RequestException as e:
                print(f"{Colors.RED}❌ FAILED with exception: {e}{Colors.ENDC}")
                failed_count += 1

    print("\n" + "="*50)
    print("Test Summary:")
    print(f"{Colors.GREEN}Passed: {passed_count}{Colors.ENDC}")
    print(f"{Colors.RED}Failed: {failed_count}{Colors.ENDC}")
    print("="*50)


def main():
    """Main execution function."""
    # Define all your test cases here
    test_cases = [
        {"name": "Health Check", "method": "GET", "endpoint": "/health", "expected_status": 200, "requires_auth": False},
        {"name": "Get Available Specialists", "method": "GET", "endpoint": "/chat/specialists", "expected_status": 200},
        {"name": "Basic Investment Committee Analysis (Auto-routing)", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "How risky is my portfolio?", "portfolio_id": 3}, "expected_status": 200},
        {"name": "Risk Analysis with Quantitative Analyst", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "Calculate my portfolio VaR", "portfolio_id": 3, "specialist": "quant"}, "expected_status": 200},
        {"name": "Portfolio Management Advice", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "Should I rebalance?", "portfolio_id": 3, "specialist": "pm"}, "expected_status": 200},
        {"name": "CIO Strategic Analysis", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "Market trends concerns?", "portfolio_id": 3, "specialist": "cio"}, "expected_status": 200},
        {"name": "Behavioral Coaching", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "I want to panic sell", "portfolio_id": 3, "specialist": "behavioral"}, "expected_status": 200},
        {"name": "Chat with Conversation History", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "Follow up on risk", "portfolio_id": 3, "chat_history": [{"role": "user", "content": "How risky?"}, {"role": "assistant", "content": "Moderate risk."}]}, "expected_status": 200},
        {"name": "Error Handling Test (Invalid Portfolio)", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "Test query", "portfolio_id": 99999}, "expected_status": 404},
        {"name": "Error Handling Test (Invalid Specialist)", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "Test query", "portfolio_id": 3, "specialist": "fake_specialist"}, "expected_status": 422},
        {"name": "Authentication Test (No Token)", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "Test without auth", "portfolio_id": 3}, "expected_status": 401, "requires_auth": False},
        {"name": "Missing Required Fields Test (No portfolio_id)", "method": "POST", "endpoint": "/chat/analyze", "payload": {"query": "test without portfolio_id"}, "expected_status": 422},
    ]

    # --- Start Execution ---
    with requests.Session() as session:
        auth_token = get_auth_token(session, BASE_URL, ADMIN_EMAIL, ADMIN_PASSWORD)
        
        if auth_token:
            run_tests(test_cases, BASE_URL, auth_token)
        else:
            print(f"{Colors.RED}\nCould not retrieve auth token. Cannot proceed with authenticated tests.{Colors.ENDC}")

if __name__ == "__main__":
    main()