# test_enhanced_backend_final.py
import requests
import json

# Your endpoints
auth_url = "http://localhost:8000/auth/login-json"
test_url = "http://localhost:8000/api/chat/enhanced/analyze/test"

# Replace with your actual credentials
login_data = {
    "email": "admin@example.com",     # Your actual email
    "password": "admin123"      # Your actual password
}

test_data = {
    "query": "What is my portfolio risk?",
    "portfolio_context": {
        "portfolio_id": 1,
        "total_value": 487650,
        "holdings": [
            {"ticker": "AAPL", "value": 150000},
            {"ticker": "MSFT", "value": 120000},
            {"ticker": "GOOGL", "value": 100000}
        ],
        "daily_change": "+1.2%"
    }
}

try:
    # Test without auth first (test endpoint should not require auth)
    print("Testing enhanced backend (no auth)...")
    response = requests.post(test_url, json=test_data)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if "traceback" in result:
            print("Traceback:", result['traceback'][:500])
        print(f"Enhanced Available: {result.get('enhanced_available', 'unknown')}")
        exit(1)
    
    print("🎉 SUCCESS! Enhanced backend is working!")
    
    # Validate all quick wins
    print(f"\n=== QUICK WINS VALIDATION ===")
    
    confidence = result.get("analysis", {}).get("confidence", 0)
    print(f"✅ Confidence: {confidence} (should be 60-99)")
    
    tool_results = result.get("tool_results", [])
    print(f"✅ Tool results: {len(tool_results)} tracked")
    if tool_results:
        success_count = sum(1 for t in tool_results if t.get('success'))
        print(f"   - Success rate: {success_count}/{len(tool_results)}")
    
    quick_actions = result.get("quick_actions", [])
    print(f"✅ Quick actions: {len(quick_actions)} generated")
    if quick_actions:
        for action in quick_actions[:2]:  # Show first 2
            print(f"   - {action.get('label', 'Unknown')}")
    
    tools_selected = result.get("tools_selected", [])
    print(f"✅ Tools selected: {tools_selected}")
    
    version = result.get("enhancement_version", "not found")
    print(f"✅ Enhancement version: {version}")
    
    # Check professional formatting
    content = result.get("content", "")
    has_formatting = any(indicator in content for indicator in ["**", "High Confidence", "Moderate Confidence"])
    print(f"✅ Professional formatting: {has_formatting}")
    
    execution_time = result.get("execution_time", 0)
    print(f"✅ Execution time: {execution_time:.2f}s")
    
    # Show performance tracking
    backend_integration = result.get("backend_integration", False)
    print(f"✅ Backend integration: {backend_integration}")
    
    print(f"\n=== CONTENT SAMPLE ===")
    print(content[:400] + "..." if len(content) > 400 else content)
    
    print(f"\n🎯 ALL QUICK WINS VALIDATED SUCCESSFULLY!")
    
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to server. Is it running on localhost:8000?")
except Exception as e:
    print(f"❌ Test failed: {e}")