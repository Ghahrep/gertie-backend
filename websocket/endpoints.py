# websocket/endpoints.py
"""
WebSocket Endpoints
==================

FastAPI WebSocket routes for real-time communication.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Optional
import json
import uuid
import logging
from datetime import datetime

from .manager import connection_manager

logger = logging.getLogger(__name__)

# Create WebSocket router
websocket_router = APIRouter()

@websocket_router.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """
    Basic WebSocket test endpoint
    
    This endpoint is for testing WebSocket functionality without authentication.
    Useful for initial setup and debugging.
    """
    connection_id = str(uuid.uuid4())
    
    try:
        # Accept connection manually to avoid auto-welcome interference
        await websocket.accept()
        
        # Store connection manually for tracking
        connection_manager.active_connections[connection_id] = websocket
        connection_manager.connection_metadata[connection_id] = {
            "user_id": None,
            "connected_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        # Send initial connection info
        await websocket.send_text(json.dumps({
            "type": "test_connection",
            "message": "WebSocket test connection established",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat(),
            "instructions": {
                "echo": "Send any message and it will be echoed back",
                "info": "Send 'info' to get connection information",
                "broadcast": "Send 'broadcast:message' to send to all test connections"
            }
        }))
        
        # Message handling loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                
                # Parse message if it's JSON, otherwise treat as plain text
                try:
                    message = json.loads(data)
                    is_json = True
                except json.JSONDecodeError:
                    message = data  # Keep as string for plain text handling
                    is_json = False
                
                # Handle JSON messages
                if is_json and isinstance(message, dict):
                    message_type = message.get("type", "unknown")
                    
                    if message_type == "ping":
                        # Respond to ping with pong
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                    elif message_type == "info":
                        # Send connection information
                        info = connection_manager.get_connection_info()
                        await websocket.send_text(json.dumps({
                            "type": "connection_info",
                            "data": info,  # This is what the test expects
                            "your_connection_id": connection_id,
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                    else:
                        # Echo the JSON message back
                        await websocket.send_text(json.dumps({
                            "type": "echo",
                            "original_message": message,
                            "connection_id": connection_id,
                            "timestamp": datetime.now().isoformat()
                        }))
                
                # Handle plain text messages
                else:
                    message_str = message if isinstance(message, str) else str(message)
                    
                    if message_str.lower() == "info":
                        # Send connection information - this is the key fix
                        info = connection_manager.get_connection_info()
                        await websocket.send_text(json.dumps({
                            "type": "connection_info",
                            "data": info,  # This is what the test expects
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                    elif message_str.startswith("broadcast:"):
                        # Broadcast message to all test connections (only for testing)
                        broadcast_content = message_str[10:]  # Remove "broadcast:" prefix
                        broadcast_msg = {
                            "type": "test_broadcast",
                            "message": broadcast_content,
                            "from_connection": connection_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        # Send to all active connections
                        for conn_id, ws in connection_manager.active_connections.items():
                            try:
                                await ws.send_text(json.dumps(broadcast_msg))
                            except:
                                pass  # Ignore send failures
                        
                    else:
                        # Echo plain text
                        await websocket.send_text(json.dumps({
                            "type": "echo",
                            "message": message_str,
                            "connection_id": connection_id,
                            "timestamp": datetime.now().isoformat()
                        }))
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket test connection {connection_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error handling message in test WebSocket {connection_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }))
                
    except Exception as e:
        logger.error(f"Error in test WebSocket endpoint {connection_id}: {e}")
    finally:
        # Clean up connection
        if connection_id in connection_manager.active_connections:
            del connection_manager.active_connections[connection_id]
        if connection_id in connection_manager.connection_metadata:
            del connection_manager.connection_metadata[connection_id]

@websocket_router.websocket("/ws/echo")
async def websocket_echo_endpoint(websocket: WebSocket):
    """
    Simple echo WebSocket endpoint
    
    Echoes back any message sent to it. Useful for testing client implementations.
    """
    connection_id = str(uuid.uuid4())
    
    try:
        # Accept connection manually to avoid auto-welcome interference
        await websocket.accept()
        
        # Send ready message
        await websocket.send_text(json.dumps({
            "type": "echo_ready",
            "message": "Echo WebSocket ready - send any message to have it echoed back",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        while True:
            try:
                data = await websocket.receive_text()
                
                # Echo the exact message back with metadata - This is what the test expects
                response = {
                    "type": "echo_response",
                    "original_data": data,  # This is what the test expects
                    "echo_timestamp": datetime.now().isoformat(),
                    "connection_id": connection_id
                }
                
                await websocket.send_text(json.dumps(response))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in echo WebSocket {connection_id}: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error setting up echo WebSocket {connection_id}: {e}")

# Health check endpoint for WebSocket status
@websocket_router.get("/ws/status")
async def websocket_status():
    """Get WebSocket service status and connection information"""
    try:
        connection_info = connection_manager.get_connection_info()
        
        return {
            "websocket_service": "operational",
            "timestamp": datetime.now().isoformat(),
            "connection_info": connection_info,
            "available_endpoints": [
                "/ws/test - Basic test endpoint with message handling",
                "/ws/echo - Simple echo endpoint",
                "/ws/status - This status endpoint"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting WebSocket status: {e}")
        return {
            "websocket_service": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }