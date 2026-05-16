#!/usr/bin/env python3
"""Benchmark client for realtime-vla server."""
import asyncio
import time
import numpy as np
import io
import sys
import msgpack

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect

def msgpack_encode(obj):
    def _encode_ext(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return msgpack.ExtType(1, buf.getvalue())
        return obj
    return msgpack.packb(obj, default=_encode_ext, strict_types=False)

def msgpack_decode(data):
    def _decode_ext(code, data):
        if code == 1:
            buf = io.BytesIO(data)
            return np.load(buf, allow_pickle=False)
        return msgpack.ExtType(code, data)
    return msgpack.unpackb(data, ext_hook=_decode_ext, raw=False)

async def test(server_url="ws://localhost:8000"):
    print(f"Connecting to {server_url}...")
    async with connect(server_url, max_size=100*1024*1024) as ws:
        # Receive metadata
        metadata = msgpack_decode(await ws.recv())
        print(f"Server metadata: {metadata}")
        
        # Create test observation
        obs = {
            "observation/images/cam_high": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/images/cam_left_wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/images/cam_right_wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.random.randn(16).astype(np.float32),
            "prompt": "put the chocolate bars in the container",
        }
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            await ws.send(msgpack_encode(obs))
            await ws.recv()
        
        # Benchmark
        print("Benchmarking (10 runs)...")
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            await ws.send(msgpack_encode(obs))
            response = msgpack_decode(await ws.recv())
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            print(f"  Run {i+1}: {times[-1]:.1f} ms")
            
        actions = response["actions"]
        print(f"\nResults:")
        print(f"  Actions shape: {np.array(actions).shape}")
        print(f"  Latency: {np.mean(times):.1f} ± {np.std(times):.1f} ms")
        print(f"  (min={np.min(times):.1f}, max={np.max(times):.1f})")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000"
    asyncio.run(test(url))
