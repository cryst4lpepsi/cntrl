"""
midi_controller.py
MIDI output to FL Studio using python-rtmidi.

Improvements:
- Rate limiting (prevents FL from choking)
- Better virtual port naming
- Safer CC sending
"""

from __future__ import annotations

import time
import rtmidi
from typing import List, Optional, Dict


class MIDIController:
    def __init__(self):
        self.midi_out = rtmidi.MidiOut()
        self.connected_port: Optional[str] = None
        self.channel = 0  # MIDI channel 1 (0-indexed)

        self._last_sent: Dict[int, float] = {}
        self._min_interval = 0.01  # 10ms per CC (safe)

    def get_available_ports(self) -> List[str]:
        ports = self.midi_out.get_ports()
        if not ports:
            return ["(Create Virtual Port)"]
        return ports

    def connect(self, port_name: str) -> bool:
        try:
            ports = self.midi_out.get_ports()

            if port_name == "(Create Virtual Port)":
                # On Windows, this won't loop into FL Studio by default.
                # Better: use loopMIDI and select that port.
                self.midi_out.open_virtual_port("GestureKnobController")
                self.connected_port = port_name
                return True

            for idx, port in enumerate(ports):
                if port_name == port:
                    self.midi_out.open_port(idx)
                    self.connected_port = port_name
                    return True

            # fallback: substring match
            for idx, port in enumerate(ports):
                if port_name in port:
                    self.midi_out.open_port(idx)
                    self.connected_port = port_name
                    return True

            return False

        except Exception as e:
            print(f"[MIDI] Connect error: {e}")
            return False

    def disconnect(self):
        try:
            if self.midi_out.is_port_open():
                self.midi_out.close_port()
        except Exception:
            pass
        self.connected_port = None

    def send_cc(self, cc_number: int, value: int):
        if not self.midi_out.is_port_open():
            return

        cc_number = int(max(0, min(127, cc_number)))
        value = int(max(0, min(127, value)))

        now = time.time()
        last_t = self._last_sent.get(cc_number, 0.0)
        if (now - last_t) < self._min_interval:
            return

        status = 0xB0 | (self.channel & 0x0F)
        msg = [status, cc_number, value]

        try:
            self.midi_out.send_message(msg)
            self._last_sent[cc_number] = now
        except Exception as e:
            print(f"[MIDI] Send error: {e}")

    def is_connected(self) -> bool:
        return self.midi_out.is_port_open()

    def close(self):
        self.disconnect()
        try:
            self.midi_out.close_port()
        except Exception:
            pass
