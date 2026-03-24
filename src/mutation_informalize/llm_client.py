"""
LLM Client: Async API calls to NVIDIA gpt-oss-120b with key rotation and rate limiting.
Supports resume via JSONL checkpointing.
"""
import asyncio
import json
import os
import time
from typing import Optional, Dict, List
from config import (
    NVIDIA_API_KEYS, BASE_URL, MODEL_NAME,
    RPM_PER_KEY, MAX_API_RETRIES, BACKOFF_BASE, MAX_TOKENS, DELAY_BETWEEN
)


class AsyncLLMClient:
    """
    Async LLM client with:
    - Dynamic Load Balancing via asyncio.Queue (Zero idle locking)
    - Per-key rate limiting guaranteed
    - Auto-retry with exponential backoff
    - REUSABLE clients per key (no connection leak)
    - Hard asyncio.wait_for timeout (guaranteed cancellation)
    """
    
    def __init__(self):
        self.keys = list(NVIDIA_API_KEYS)
        self.delay = DELAY_BETWEEN  # seconds between calls per key
        self._last_call_time = {i: 0.0 for i in range(len(self.keys))}
        self.key_queue = None
        self._clients = {}  # Tái sử dụng client theo key_idx
    
    def _get_client(self, key_idx: int, api_key: str):
        """Lấy hoặc tạo client cho key, tái sử dụng để tránh rò rỉ kết nối."""
        from openai import AsyncOpenAI
        if key_idx not in self._clients:
            self._clients[key_idx] = AsyncOpenAI(
                base_url=BASE_URL, 
                api_key=api_key,
                timeout=60.0,  # httpx-level timeout
            )
        return self._clients[key_idx]

    async def _make_api_call(self, client, system_prompt, user_prompt, temperature):
        """Thực hiện 1 lần gọi API thuần túy (không có logic retry)."""
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=0.9,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion

    async def call(self, system_prompt: str, user_prompt: str,
                   temperature: float = 0.7) -> Optional[str]:
        """
        Make a single API call with dynamic key pool allocation.
        Uses asyncio.wait_for() for HARD timeout guarantee.
        """
        # Khởi tạo Queue lười (Lazy init) bên trong event loop
        if self.key_queue is None:
            self.key_queue = asyncio.Queue()
            for idx, key in enumerate(self.keys):
                self.key_queue.put_nowait((idx, key))
        
        for attempt in range(MAX_API_RETRIES + 1):
            # Lấy key đang rảnh rỗi ra khỏi hàng đợi (Load balancing động)
            key_idx, api_key = await self.key_queue.get()
            
            try:
                # Đảm bảo khoảng cách thời gian rate limit trên chính key này
                now = time.time()
                elapsed = now - self._last_call_time[key_idx]
                if elapsed < self.delay:
                    await asyncio.sleep(self.delay - elapsed)
                
                client = self._get_client(key_idx, api_key)
                
                # BẢN VÁ CHỐNG TREO: asyncio.wait_for() cưỡng chế hủy sau 90s
                # Đây là lớp bảo vệ cuối cùng, KHÔNG THỂ bị vượt qua
                completion = await asyncio.wait_for(
                    self._make_api_call(client, system_prompt, user_prompt, temperature),
                    timeout=90.0
                )
                
                self._last_call_time[key_idx] = time.time()
                
                content = completion.choices[0].message.content
                if not content or not content.strip():
                    raise RuntimeError("empty_response")
                
                # Thành công thì trả key ngay lập tức và return
                self.key_queue.put_nowait((key_idx, api_key))
                return content.strip()
                
            except KeyboardInterrupt:
                self.key_queue.put_nowait((key_idx, api_key))
                raise
            except asyncio.TimeoutError:
                # Timeout cứng: XÓA client cũ ngay lập tức (KHÔNG await close() vì nó cũng treo!)
                print(f"[!] TIMEOUT 90s (key={key_idx}), destroying stuck connection...")
                self._last_call_time[key_idx] = time.time()
                # Xóa reference, để garbage collector dọn dẹp socket
                self._clients.pop(key_idx, None)
                self.key_queue.put_nowait((key_idx, api_key))
                
                if attempt == MAX_API_RETRIES:
                    print(f"[!] API failed after {MAX_API_RETRIES} retries (timeout)")
                    return None
                    
                await asyncio.sleep(2)
            except Exception as e:
                # Lỗi thì cập nhật thời gian để key nghỉ ngơi, sau đó TRẢ KEY NGAY LẬP TỨC vào Queue!
                self._last_call_time[key_idx] = time.time()
                self.key_queue.put_nowait((key_idx, api_key))
                
                if attempt == MAX_API_RETRIES:
                    print(f"[!] API failed after {MAX_API_RETRIES} retries: {str(e)[:60]}")
                    return None
                
                # Rút ngắn thời gian chờ khi lỗi: dùng 2s cơ sở thay vì 8s để tránh treo luồng quá lâu
                wait = 2 * (2 ** attempt)
                print(f"[!] API error (key={key_idx}): {str(e)[:40]}... task sleeping {wait}s")
                await asyncio.sleep(wait)
        
        return None


# ─── Checkpoint/Resume Helper ───

def load_checkpoint(filepath: str, clean_failed: bool = False) -> dict:
    """
    Load processed items from JSONL checkpoint file.
    Returns dict: {index: record}
    If clean_failed is True, it will remove all records with "status": "failed" from the file,
    and only return the successful indices.
    """
    processed = {}
    if not os.path.exists(filepath):
        return processed
    
    valid_lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                idx = rec.get("index", "")
                
                # Bỏ qua các record failed nếu clean_failed = True
                if clean_failed and rec.get("status") == "failed":
                    continue
                    
                if idx != "":
                    processed[idx] = rec
                valid_lines.append(line)
            except:
                pass
                
    if clean_failed:
        with open(filepath, "w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")
                
    return processed


def append_checkpoint(filepath: str, idx: int, record: dict):
    """Append a single record to JSONL checkpoint."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_progress(done: int, total: int, desc: str = "", extra: str = ""):
    """Print a progress bar."""
    pct = done / total * 100 if total else 0
    bar_len = 30
    filled = int(bar_len * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    line = f"\r  {desc} |{bar}| {done}/{total} ({pct:.1f}%) {extra}"
    print(f"{line:<100s}", end="", flush=True)
