import asyncio
import http
import logging
import time
import traceback

import cv2
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _decode_jpeg_images(obs: dict) -> dict:
    """Decode JPEG-encoded images in observation dict.
    
    Images can be either:
    - bytes (JPEG encoded) -> decode to [C, H, W] uint8
    - numpy array [C, H, W] (raw) -> pass through
    """
    if "images" not in obs:
        return obs
    
    images = obs["images"]
    decoded = {}
    for key, img in images.items():
        if isinstance(img, (bytes, bytearray)):
            # JPEG bytes -> decode to numpy
            arr = np.frombuffer(img, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                # Convert BGR to RGB and transpose to [C, H, W]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded[key] = np.transpose(rgb, (2, 0, 1))
            else:
                logger.warning(f"Failed to decode JPEG for image '{key}'")
                decoded[key] = np.zeros((3, 224, 224), dtype=np.uint8)
        else:
            # Already numpy array, pass through
            decoded[key] = img
    
    obs["images"] = decoded
    return obs


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                
                recv_time = time.monotonic()
                raw_data = await websocket.recv()
                recv_time = time.monotonic() - recv_time
                
                unpack_time = time.monotonic()
                obs = msgpack_numpy.unpackb(raw_data)
                unpack_time = time.monotonic() - unpack_time
                
                # Decode JPEG-encoded images if present
                decode_time = time.monotonic()
                obs = _decode_jpeg_images(obs)
                decode_time = time.monotonic() - decode_time

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                
                # Log detailed timing every 50 requests
                if not hasattr(self, '_req_count'):
                    self._req_count = 0
                self._req_count += 1
                if self._req_count % 50 == 0:
                    print(f"[SERVER-TIMING] recv={recv_time*1000:.1f}ms unpack={unpack_time*1000:.1f}ms "
                          f"decode={decode_time*1000:.1f}ms infer={infer_time*1000:.1f}ms "
                          f"data_size={len(raw_data)/1024:.1f}KB", flush=True)

                # Log gripper values for debugging (indices 7=left, 15=right)
                if "actions" in action:
                    actions = action["actions"]
                    if hasattr(actions, 'shape') and len(actions.shape) >= 1:
                        first_action = actions[0] if len(actions.shape) > 1 else actions
                        if len(first_action) >= 16:
                            logger.info(f"[GRIPPER] left={first_action[7]:.3f}, right={first_action[15]:.3f}")

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
