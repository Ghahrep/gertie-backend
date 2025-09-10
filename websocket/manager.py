# websocket/manager.py
"""
WebSocket Connection Manager
===========================

Handles WebSocket connections, message broadcasting, and connection lifecycle.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import json
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        # Store active connections by connection_id
        self.active_connections: Dict[str, WebSocket] = {}
        # Group connections by user_id for user-specific broadcasting
        self.user_connections: Dict[int, Set[str]] = {}
        # Store connection metadata
        self.connection_metadata: Dict[str, Dict] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: int = None):
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Store connection
            self.active_connections[connection_id] = websocket
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "connected_at": datetime.now(),
                "last_activity": datetime.now()
            }
            
            # Track user connections
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            logger.info(f"WebSocket connection {connection_id} established for user {user_id}")
            
            # Send welcome message
            await self.send_personal_message({
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat(),
                "message": "WebSocket connection established"
            }, connection_id)
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection {connection_id}: {e}")
            await self.disconnect(connection_id)
    
    async def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            try:
                # Get user_id before removing metadata
                user_id = self.connection_metadata.get(connection_id, {}).get("user_id")
                
                # Remove from active connections
                del self.active_connections[connection_id]
                del self.connection_metadata[connection_id]
                
                # Remove from user connections
                if user_id and user_id in self.user_connections:
                    self.user_connections[user_id].discard(connection_id)
                    if not self.user_connections[user_id]:
                        del self.user_connections[user_id]
                
                logger.info(f"WebSocket connection {connection_id} disconnected for user {user_id}")
                
            except Exception as e:
                logger.error(f"Error during WebSocket disconnection {connection_id}: {e}")
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send a message to a specific connection"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message))
                
                # Update last activity
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_activity"] = datetime.now()
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket {connection_id} disconnected during message send")
                await self.disconnect(connection_id)
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                await self.disconnect(connection_id)
    
    async def send_user_message(self, message: dict, user_id: int):
        """Send a message to all connections for a specific user"""
        if user_id in self.user_connections:
            connection_ids = list(self.user_connections[user_id])  # Copy to avoid modification during iteration
            for connection_id in connection_ids:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: dict):
        """Send a message to all active connections"""
        connection_ids = list(self.active_connections.keys())  # Copy to avoid modification during iteration
        for connection_id in connection_ids:
            await self.send_personal_message(message, connection_id)
    
    async def send_to_connections(self, message: dict, connection_ids: List[str]):
        """Send a message to specific connections"""
        for connection_id in connection_ids:
            await self.send_personal_message(message, connection_id)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.active_connections)
    
    def get_user_connection_count(self, user_id: int) -> int:
        """Get number of connections for a specific user"""
        return len(self.user_connections.get(user_id, set()))
    
    def get_active_users(self) -> List[int]:
        """Get list of user IDs with active connections"""
        return list(self.user_connections.keys())
    
    def get_connection_info(self) -> Dict:
        """Get information about all active connections"""
        # Fixed: Ensure consistent structure and handle empty connections
        oldest_connection = None
        if self.connection_metadata:
            oldest_timestamp = min([
                metadata["connected_at"] 
                for metadata in self.connection_metadata.values()
            ])
            oldest_connection = oldest_timestamp.isoformat()
        
        return {
            "total_connections": self.get_connection_count(),
            "active_users": len(self.user_connections),
            "connections_by_user": {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            },
            "oldest_connection": oldest_connection
        }

# Global connection manager instance
connection_manager = ConnectionManager()