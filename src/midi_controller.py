import time
import rtmidi


class MIDIController:
    def __init__(self):
        self.midi_out = rtmidi.MidiOut()
        self.connected_port = None

        # Prevent spamming FL Studio - REDUCED for better responsiveness
        self.last_cc = {}  # (channel, cc) -> value
        self.last_sent_time = {}  # (channel, cc) -> time
        self.min_interval = 0.005  # 5ms per CC max (200 fps)

    def get_available_ports(self):
        return list(self.midi_out.get_ports())

    def connect(self, port_name: str) -> bool:
        ports = self.midi_out.get_ports()

        target_index = None
        for i, p in enumerate(ports):
            if p.strip() == port_name.strip():
                target_index = i
                break

        if target_index is None:
            print(f"[MIDI] Port not found: {port_name}")
            print("[MIDI] Available ports:", ports)
            return False

        try:
            self.midi_out.open_port(target_index)
            self.connected_port = port_name
            print(f"[MIDI] Connected to: {port_name}")
            return True
        except Exception as e:
            print("[MIDI] Failed to open port:", e)
            return False

    def is_connected(self) -> bool:
        return self.connected_port is not None

    def send_cc(self, cc: int, value: int, channel: int = 0):
        if not self.is_connected():
            return

        cc = int(max(0, min(127, cc)))
        value = int(max(0, min(127, value)))
        channel = int(max(0, min(15, channel)))

        key = (channel, cc)
        now = time.time()  # FIXED: Moved BEFORE first use

        # FIXED: Allow resending same value if enough time passed
        if self.last_cc.get(key) == value:
            last_t = self.last_sent_time.get(key, 0.0)
            # Only skip if recently sent the exact same value
            if (now - last_t) < 0.1:  # 100ms window
                return

        # Rate limit per CC
        last_t = self.last_sent_time.get(key, 0.0)
        if (now - last_t) < self.min_interval:
            return

        self.last_cc[key] = value
        self.last_sent_time[key] = now

        msg = [0xB0 | channel, cc, value]
        try:
            self.midi_out.send_message(msg)
            # print(f"[MIDI] Sent CC{cc} = {value}")  # Uncomment for debug
        except Exception as e:
            print(f"[MIDI] Send error: {e}")

    def reset_cache(self):
        """Call when tracking starts to avoid old CC values blocking first send."""
        self.last_cc.clear()
        self.last_sent_time.clear()
        print("[MIDI] Cache reset")

    def close(self):
        try:
            self.midi_out.close_port()
        except Exception:
            pass
        self.connected_port = None
        self.reset_cache()