"""
WebSocket route for real-time question processing updates.

This module provides WebSocket endpoints for clients to receive
real-time updates about question processing status, agent activities,
and backtracking events.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Set
import json
import asyncio
import logging
from datetime import datetime
import redis.asyncio as redis
from collections import defaultdict
import uuid

from reagent.models import Message, MessageType
from api.schemas.websocket import (
    WSMessage,
    WSMessageType,
    WSQuestionStatus,
    WSAgentActivity,
    WSBacktrackingEvent,
    WSMetricsUpdate
)
from api.deps import get_redis_client, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.question_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.client_questions: Dict[str, Set[str]] = defaultdict(set)
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")
        
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Clean up subscriptions
            questions = self.client_questions.get(client_id, set())
            for question_id in questions:
                self.question_subscriptions[question_id].discard(client_id)
                
            if client_id in self.client_questions:
                del self.client_questions[client_id]
                
            logger.info(f"WebSocket client {client_id} disconnected")
    
    def subscribe_to_question(self, client_id: str, question_id: str):
        """Subscribe a client to updates for a specific question."""
        self.question_subscriptions[question_id].add(client_id)
        self.client_questions[client_id].add(question_id)
        logger.debug(f"Client {client_id} subscribed to question {question_id}")
    
    def unsubscribe_from_question(self, client_id: str, question_id: str):
        """Unsubscribe a client from question updates."""
        self.question_subscriptions[question_id].discard(client_id)
        self.client_questions[client_id].discard(question_id)
        logger.debug(f"Client {client_id} unsubscribed from question {question_id}")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client."""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_question(self, message: str, question_id: str):
        """Broadcast a message to all clients subscribed to a question."""
        client_ids = self.question_subscriptions.get(question_id, set())
        
        # Send to all subscribed clients
        tasks = []
        for client_id in client_ids:
            if client_id in self.active_connections:
                tasks.append(
                    self.send_personal_message(message, client_id)
                )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_to_all(self, message: str):
        """Broadcast a message to all connected clients."""
        tasks = []
        for client_id in list(self.active_connections.keys()):
            tasks.append(
                self.send_personal_message(message, client_id)
            )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# Global connection manager instance
manager = ConnectionManager()

# Background task to process Redis messages
async def redis_message_processor(redis_client: redis.Redis):
    """Process messages from Redis and broadcast to WebSocket clients."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(
        "reagent:questions:*",
        "reagent:agents:*",
        "reagent:backtracking:*",
        "reagent:metrics"
    )
    
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"].decode()
                data = json.loads(message["data"])
                
                # Route message based on channel
                if channel.startswith("reagent:questions:"):
                    question_id = channel.split(":")[-1]
                    await handle_question_update(question_id, data)
                    
                elif channel.startswith("reagent:agents:"):
                    await handle_agent_update(data)
                    
                elif channel.startswith("reagent:backtracking:"):
                    await handle_backtracking_update(data)
                    
                elif channel == "reagent:metrics":
                    await handle_metrics_update(data)
                    
    except asyncio.CancelledError:
        await pubsub.unsubscribe()
        await pubsub.close()
        raise
    except Exception as e:
        logger.error(f"Error in Redis message processor: {e}")

async def handle_question_update(question_id: str, data: Dict):
    """Handle question status updates."""
    try:
        ws_message = WSMessage(
            type=WSMessageType.QUESTION_UPDATE,
            data=WSQuestionStatus(
                question_id=question_id,
                status=data.get("status", "processing"),
                progress=data.get("progress", 0),
                current_phase=data.get("current_phase"),
                sub_questions=data.get("sub_questions", []),
                partial_answer=data.get("partial_answer"),
                final_answer=data.get("final_answer"),
                confidence=data.get("confidence"),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
            )
        )
        
        await manager.broadcast_to_question(
            ws_message.model_dump_json(),
            question_id
        )
        
    except Exception as e:
        logger.error(f"Error handling question update: {e}")

async def handle_agent_update(data: Dict):
    """Handle agent activity updates."""
    try:
        ws_message = WSMessage(
            type=WSMessageType.AGENT_UPDATE,
            data=WSAgentActivity(
                agent_id=data["agent_id"],
                action=data["action"],
                status=data.get("status", "active"),
                processing_time=data.get("processing_time"),
                token_usage=data.get("token_usage"),
                message_content=data.get("message_content"),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
            )
        )
        
        # Broadcast to all connected clients for now
        # In production, might want to filter by question
        await manager.broadcast_to_all(ws_message.model_dump_json())
        
    except Exception as e:
        logger.error(f"Error handling agent update: {e}")

async def handle_backtracking_update(data: Dict):
    """Handle backtracking event updates."""
    try:
        ws_message = WSMessage(
            type=WSMessageType.BACKTRACKING,
            data=WSBacktrackingEvent(
                event_id=data.get("event_id", str(uuid.uuid4())),
                type=data["type"],  # local or global
                agent_id=data.get("agent_id"),
                question_id=data.get("question_id"),
                checkpoint_id=data.get("checkpoint_id"),
                reason=data["reason"],
                affected_agents=data.get("affected_agents", []),
                rollback_depth=data.get("rollback_depth", 1),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
            )
        )
        
        # Broadcast to question subscribers if question_id present
        if data.get("question_id"):
            await manager.broadcast_to_question(
                ws_message.model_dump_json(),
                data["question_id"]
            )
        else:
            # Global backtracking affects all
            await manager.broadcast_to_all(ws_message.model_dump_json())
        
    except Exception as e:
        logger.error(f"Error handling backtracking update: {e}")

async def handle_metrics_update(data: Dict):
    """Handle system metrics updates."""
    try:
        ws_message = WSMessage(
            type=WSMessageType.METRICS,
            data=WSMetricsUpdate(
                active_questions=data.get("active_questions", 0),
                total_processed=data.get("total_processed", 0),
                success_rate=data.get("success_rate", 0.0),
                avg_processing_time=data.get("avg_processing_time", 0.0),
                active_agents=data.get("active_agents", {}),
                backtracking_count=data.get("backtracking_count", 0),
                token_usage_total=data.get("token_usage_total", 0),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
            )
        )
        
        # Broadcast metrics to all connected clients
        await manager.broadcast_to_all(ws_message.model_dump_json())
        
    except Exception as e:
        logger.error(f"Error handling metrics update: {e}")

# WebSocket endpoint with authentication
security = HTTPBearer(auto_error=False)

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = None,
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """
    WebSocket endpoint for real-time updates.
    
    Clients can:
    - Subscribe to specific question updates
    - Receive agent activity notifications
    - Monitor backtracking events
    - Get system metrics updates
    
    Authentication via query parameter: ws://host/ws?token=YOUR_TOKEN
    """
    client_id = str(uuid.uuid4())
    
    # Verify authentication token if provided
    if token:
        try:
            # In production, verify JWT token or API key
            # For now, simple check
            if not token.startswith("Bearer "):
                await websocket.close(code=1008, reason="Invalid token format")
                return
        except Exception:
            await websocket.close(code=1008, reason="Authentication failed")
            return
    
    await manager.connect(websocket, client_id)
    
    # Send welcome message
    welcome_msg = WSMessage(
        type=WSMessageType.SYSTEM,
        data={
            "message": "Connected to ReAgent WebSocket",
            "client_id": client_id,
            "version": "1.0.0"
        }
    )
    await manager.send_personal_message(
        welcome_msg.model_dump_json(),
        client_id
    )
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                command = message.get("command")
                
                if command == "subscribe":
                    question_id = message.get("question_id")
                    if question_id:
                        manager.subscribe_to_question(client_id, question_id)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "subscription_confirmed",
                                "question_id": question_id
                            }),
                            client_id
                        )
                
                elif command == "unsubscribe":
                    question_id = message.get("question_id")
                    if question_id:
                        manager.unsubscribe_from_question(client_id, question_id)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "unsubscription_confirmed",
                                "question_id": question_id
                            }),
                            client_id
                        )
                
                elif command == "ping":
                    await manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                        client_id
                    )
                    
                else:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": f"Unknown command: {command}"
                        }),
                        client_id
                    )
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }),
                    client_id
                )
            except Exception as e:
                logger.error(f"Error processing client message: {e}")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Internal server error"
                    }),
                    client_id
                )
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status."""
    return {
        "active_connections": len(manager.active_connections),
        "active_questions": len(manager.question_subscriptions),
        "clients": list(manager.active_connections.keys())
    }
