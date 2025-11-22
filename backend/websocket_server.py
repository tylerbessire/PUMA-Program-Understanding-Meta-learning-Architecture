"""
WebSocket Server

Real-time communication between consciousness and GUI.
"""

from typing import List, Dict, Any
import json
import asyncio

# Conditional import - will work when FastAPI is installed
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not installed. WebSocket server will not be available.")


class ConsciousnessWebSocketManager:
    """
    Manages WebSocket connections and broadcasts updates to GUI.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.consciousness = None  # Will be set by main system

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"📡 WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"📡 WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """
        Send updates to all connected clients.

        Args:
            event_type: Type of event ('state_change', 'new_episode', etc.)
            data: Event data
        """
        message = {
            'type': event_type,
            'data': data,
            'timestamp': data.get('timestamp', '')
        }

        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def handle_client_message(self, message: Dict[str, Any]):
        """
        Handle incoming message from GUI client.

        Args:
            message: Client message with 'type' and data
        """
        msg_type = message.get('type')

        if msg_type == 'user_interrupt':
            if self.consciousness:
                self.consciousness.state_machine.interrupt_flag = True

        elif msg_type == 'approve_modification':
            mod_id = message.get('modificationId')
            if self.consciousness:
                self.consciousness.shop_modification.approve_modification(mod_id)

        elif msg_type == 'ask_question':
            question = message.get('question')
            if self.consciousness and question:
                self.consciousness.curiosity.add_questions([question])

        elif msg_type == 'get_status':
            # Send current status
            await self.broadcast('status', self._get_current_status())

    def _get_current_status(self) -> Dict[str, Any]:
        """Get current consciousness status"""
        if not self.consciousness:
            return {'status': 'not_initialized'}

        return {
            'state': self.consciousness.state_machine.current_state.value,
            'total_episodes': self.consciousness.memory.count_total_episodes(),
            'open_questions': len(self.consciousness.curiosity.open_questions),
            'active_goals': len(self.consciousness.goals.active_goals),
            'atoms': self.consciousness.atomspace.count_atoms(),
            'uptime': self.consciousness.self_model.temporal_self.get_lifetime_duration()
        }


# FastAPI app (only if FastAPI is available)
if FASTAPI_AVAILABLE:
    app = FastAPI(title="PUMA Consciousness Backend")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global WebSocket manager
    ws_manager = ConsciousnessWebSocketManager()

    @app.websocket("/consciousness")
    async def consciousness_stream(websocket: WebSocket):
        """WebSocket endpoint for consciousness stream"""
        await ws_manager.connect(websocket)

        try:
            while True:
                # Receive commands from GUI
                data = await websocket.receive_json()
                await ws_manager.handle_client_message(data)

        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "PUMA Consciousness Backend",
            "status": "running",
            "endpoints": {
                "websocket": "/consciousness"
            }
        }

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "active_connections": len(ws_manager.active_connections)
        }

else:
    # Placeholder if FastAPI not available
    app = None
    ws_manager = None


def run_server(host: str = "localhost", port: int = 8000):
    """
    Run WebSocket server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not installed. Cannot start server.")
        print("Install with: pip install fastapi uvicorn")
        return

    try:
        import uvicorn
        print(f"🚀 Starting PUMA Backend Server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")


if __name__ == "__main__":
    run_server()
