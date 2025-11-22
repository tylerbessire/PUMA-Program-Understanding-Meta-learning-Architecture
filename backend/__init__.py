"""
Backend Server

WebSocket server for real-time communication with GUI.
"""

from .websocket_server import ConsciousnessWebSocketManager, app

__all__ = ['ConsciousnessWebSocketManager', 'app']
