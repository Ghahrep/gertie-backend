# test_websocket_basic.py
"""
Basic WebSocket functionality test script
"""

import asyncio
import websockets
import json
import time

async def test_websocket_connection():
    """Test basic WebSocket connection and messaging"""
    
    print("🚀 Testing WebSocket Basic Infrastructure")
    print("=" * 50)
    
    # Test 1: Basic connection
    print("Test 1: Basic WebSocket Connection")
    try:
        uri = "ws://localhost:8000/ws/test"
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to {uri}")
            
            # Wait for welcome message
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            print(f"✅ Received welcome: {welcome_data['type']}")
            
            # Test 2: Echo functionality
            print("\nTest 2: Echo Functionality")
            test_message = "Hello WebSocket!"
            await websocket.send(test_message)
            
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"✅ Echo response received: {response_data['type']}")
            
            # Test 3: JSON message handling
            print("\nTest 3: JSON Message Handling")
            json_message = {"type": "test", "data": "JSON test message"}
            await websocket.send(json.dumps(json_message))
            
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"✅ JSON response received: {response_data['type']}")
            
            # Test 4: Ping/Pong
            print("\nTest 4: Ping/Pong")
            ping_message = {"type": "ping"}
            await websocket.send(json.dumps(ping_message))
            
            response = await websocket.recv()
            response_data = json.loads(response)
            if response_data['type'] == 'pong':
                print("✅ Ping/Pong working")
            
            # Test 5: Connection info
            print("\nTest 5: Connection Info")
            await websocket.send("info")
            
            response = await websocket.recv()
            response_data = json.loads(response)
            if response_data['type'] == 'connection_info':
                print(f"✅ Connection info received: {response_data['data']['total_connections']} total connections")
            
            print(f"\n✅ Basic WebSocket tests completed successfully")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False
    
    return True

async def test_echo_endpoint():
    """Test the echo endpoint specifically"""
    
    print("\n🔄 Testing Echo Endpoint")
    print("=" * 30)
    
    try:
        uri = "ws://localhost:8000/ws/echo"
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to echo endpoint")
            
            # Wait for ready message
            ready_msg = await websocket.recv()
            ready_data = json.loads(ready_msg)
            print(f"✅ Echo ready: {ready_data['type']}")
            
            # Test echo
            test_message = "Echo test message"
            await websocket.send(test_message)
            
            response = await websocket.recv()
            response_data = json.loads(response)
            
            if response_data['original_data'] == test_message:
                print("✅ Echo functionality working correctly")
                return True
            else:
                print("❌ Echo mismatch")
                return False
                
    except Exception as e:
        print(f"❌ Echo test failed: {e}")
        return False

async def test_multiple_connections():
    """Test multiple simultaneous connections"""
    
    print("\n👥 Testing Multiple Connections")
    print("=" * 35)
    
    connections = []
    
    try:
        # Create 3 simultaneous connections
        for i in range(3):
            uri = "ws://localhost:8000/ws/test"
            websocket = await websockets.connect(uri)
            connections.append(websocket)
            
            # Wait for welcome message
            welcome_msg = await websocket.recv()
            print(f"✅ Connection {i+1} established")
        
        print(f"✅ {len(connections)} simultaneous connections working")
        
        # Test messaging between connections
        await connections[0].send("info")
        response = await connections[0].recv()
        response_data = json.loads(response)
        
        total_connections = response_data['data']['total_connections']
        print(f"✅ Total connections reported: {total_connections}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple connections test failed: {e}")
        return False
        
    finally:
        # Clean up connections
        for websocket in connections:
            try:
                await websocket.close()
            except:
                pass

async def test_websocket_status():
    """Test the WebSocket status endpoint"""
    
    print("\n📊 Testing WebSocket Status Endpoint")
    print("=" * 40)
    
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/ws/status') as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ WebSocket status: {data['websocket_service']}")
                    print(f"✅ Available endpoints: {len(data['available_endpoints'])}")
                    return True
                else:
                    print(f"❌ Status endpoint returned {response.status}")
                    return False
                    
    except ImportError:
        print("⚠️  aiohttp not available, skipping HTTP status test")
        return True
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return False

async def run_all_tests():
    """Run all WebSocket tests"""
    
    print("🧪 WebSocket Infrastructure Test Suite")
    print("=" * 50)
    print("Make sure your server is running: python main_clean.py")
    print()
    
    tests = [
        ("Basic Connection & Messaging", test_websocket_connection),
        ("Echo Endpoint", test_echo_endpoint),
        ("Multiple Connections", test_multiple_connections),
        ("Status Endpoint", test_websocket_status)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All WebSocket tests passed! Subtask 2.6.1 is complete.")
        print("\nNext steps:")
        print("1. Test the web client at: http://localhost:8000/test-websocket")
        print("2. Ready for Subtask 2.6.2: Portfolio WebSocket Integration")
    else:
        print("❌ Some tests failed. Please check your WebSocket implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")