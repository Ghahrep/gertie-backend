# debug_websocket_test.py
"""
Debug script to see exact message structures
"""

import asyncio
import websockets
import json

async def debug_echo_endpoint():
    """Debug the echo endpoint to see exact message structure"""
    
    print("🔍 Debugging Echo Endpoint")
    print("=" * 30)
    
    try:
        uri = "ws://localhost:8000/ws/echo"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to echo endpoint")
            
            # Wait for ready message
            ready_msg = await websocket.recv()
            ready_data = json.loads(ready_msg)
            print(f"Ready message: {ready_data}")
            
            # Send test message
            test_message = "Echo test message"
            print(f"\nSending: '{test_message}'")
            await websocket.send(test_message)
            
            # Receive response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"Received response: {response_data}")
            print(f"Response keys: {list(response_data.keys())}")
            
            # Check if 'original_data' key exists
            if 'original_data' in response_data:
                print(f"✅ original_data found: {response_data['original_data']}")
                if response_data['original_data'] == test_message:
                    print("✅ Echo content matches")
                else:
                    print(f"❌ Echo mismatch: expected '{test_message}', got '{response_data['original_data']}'")
            else:
                print(f"❌ 'original_data' key missing from response")
                print(f"Available keys: {list(response_data.keys())}")
                
    except Exception as e:
        print(f"❌ Debug failed: {e}")

async def debug_connection_info():
    """Debug the connection info to see exact structure"""
    
    print("\n🔍 Debugging Connection Info")
    print("=" * 35)
    
    try:
        uri = "ws://localhost:8000/ws/test"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to test endpoint")
            
            # Wait for welcome message
            welcome_msg = await websocket.recv()
            print("Welcome message received")
            
            # Request connection info
            print("\nSending 'info' request...")
            await websocket.send("info")
            
            # Receive response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"Info response: {response_data}")
            print(f"Response keys: {list(response_data.keys())}")
            
            # Check if 'data' key exists
            if 'data' in response_data:
                print(f"✅ 'data' key found: {response_data['data']}")
                data_content = response_data['data']
                if 'total_connections' in data_content:
                    print(f"✅ total_connections: {data_content['total_connections']}")
                else:
                    print("❌ 'total_connections' missing from data")
            else:
                print(f"❌ 'data' key missing from response")
                print(f"Available keys: {list(response_data.keys())}")
                
    except Exception as e:
        print(f"❌ Debug failed: {e}")

async def run_debug():
    """Run all debug tests"""
    
    print("🐛 WebSocket Debug Suite")
    print("=" * 25)
    print("This will show exact message structures to identify test failures\n")
    
    await debug_echo_endpoint()
    await debug_connection_info()
    
    print("\n" + "=" * 50)
    print("Debug complete. Check the message structures above.")

if __name__ == "__main__":
    try:
        asyncio.run(run_debug())
    except KeyboardInterrupt:
        print("\n⏹️ Debug interrupted by user")
    except Exception as e:
        print(f"\n❌ Debug suite failed: {e}")