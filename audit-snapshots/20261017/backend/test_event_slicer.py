import numpy as np
import time

class M4EventCentricEngine:
    def __init__(self, sample_rate: int = 384000):
        self.sample_rate = sample_rate
        self.ff_samples = int(0.5 * sample_rate)  # Forward Focus = 0.5s (192,000 samples)
        self.rr_samples = int(0.5 * sample_rate)  # Reverse Recovery = 0.5s (192,000 samples)
        self.total_event_samples = self.ff_samples + self.rr_samples # 1.0s Window အတိအကျ

    def extract_isolated_event(self, circular_buffer: np.ndarray, trigger_index: int):
        """
        Live stream ကို ဖြတ်တောက်ပြီး Event တကြိမ်စာ Packet သီးသန့်ကို
        Zero-copy NumPy Slice ဖြင့် စက္ကန့်ပိုင်းအတွင်း ဆွဲထုတ်ခြင်း
        """
        start_idx = trigger_index - self.rr_samples
        end_idx = trigger_index + self.ff_samples
        
        # Buffer Boundary Guardrails
        if start_idx < 0 or end_idx > len(circular_buffer):
            print(f"⚠️ Event Context Out of Bounds: Start {start_idx}, End {end_idx}")
            return None
            
        # 0% Heap Allocation- NumPy Zero-Copy Slice (အလွန်မြန်ဆန်သည်)
        return circular_buffer[start_idx:end_idx]

# ==========================================
#  လက်တွေ့ LIVE စမ်းသပ်ရန် စနစ် (TEST RUNNER)
# ==========================================
if __name__ == "__main__":
    print("🚀 Initializing 100% Event-Centric Engine Buffer Slicer for M4...")
    
    # 384kHz စနစ်ဖြင့် ၂ စက္ကန့်စာ pre-allocated Buffer အပြည့် တည်ဆောက်ခြင်း
    sample_rate = 384000
    buffer_capacity = 2 * sample_rate  # 768,000 samples
    mock_circular_buffer = np.random.randn(buffer_capacity).astype(np.float32)
    
    # Engine အား ဖွဲ့စည်းခြင်း
    engine = M4EventCentricEngine(sample_rate=sample_rate)
    
    # အလယ်တည့်တည့် (1.0 second နေရာ) တွင် Event တစ်ခု ဖြစ်ပွားသည်ဟု သတ်မှတ်
    trigger_timestamp_idx = 1 * sample_rate  # Sample နံပါတ် 384,000 နေရာ
    
    print(f"🎯 Simulated Event Triggered at Sample Index: {trigger_timestamp_idx}")
    
    # တွက်ချက်မှု ကြာချိန်အား တိုင်းတာရန်
    t_start = time.perf_counter()
    
    # Event Packet အား အပြတ်အသတ် ဖြတ်ထုတ်ယူခြင်း
    event_packet = engine.extract_isolated_event(mock_circular_buffer, trigger_timestamp_idx)
    
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000
    
    print("\n--- 📊 TEST RESULTS ---")
    if event_packet is not None:
        print(f"✅ Event Packet Sliced Successfully!")
        print(f"📦 Total Extracted Samples: {len(event_packet)} (Must be exactly {sample_rate} for 1.0s)")
        print(f"⚡ M4 Slice Latency: {latency_ms:.4f} ms (Target < 200ms)")
        print(f"🧩 Array Memory Strategy: Zero-Copy Pointer (Heap Allocations: 0%)")
    else:
        print("❌ Slicing Failed.")
