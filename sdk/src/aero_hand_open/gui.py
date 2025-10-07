#!/usr/bin/env python3
"""
firmware_gui.py - Minimal GUI for TetherIA Aero Hand (16-byte serial protocol)

A simple Tkinter GUI to control the TetherIA Aero Hand via serial port:
- "Start Homing": sends a HOMING command 16 byte packet to the ESP.
- "Set-ID Servo": asks for an integer ID (0..250) and sends a REID command + the integer.
- "Trim Servo": asks for the servo id and the degrees +360/-360.
- "Upload Firmware" (select .bin and flash with esptool).
- 7 sliders (0..65535) to control joints  by sending 16 Bytes CTRL_POS Command.
- RX log window to show incoming parsed serial data and status messages.
- Status bar for connection status and info.
- Adjustable TX rate (default 40 Hz).
- Auto-detects merged vs app-only .bin files for flashing.
- Handles esptool installation if missing.
- Uses pyserial for serial communication.
"""
import sys
import os
import threading
import time
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

from serial.tools import list_ports

from aero_hand_open.aero_hand import AeroHand
# ---- operation codes ------------
HOMING_MODE = 0x01
SET_ID_MODE = 0x03
TRIM_MODE   = 0x04

CTRL_POS = 0x11

GET_ALL  = 0x21
GET_POS  = 0x22
GET_VEL  = 0x23
GET_CURR = 0x24
GET_TEMP = 0x25
# ---- GUI ---------------------------------------------------------------------
BAUDS = [
    9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600
]

SLIDER_LABELS = [
    "thumb_abduction",
    "thumb_flex",
    "thumb_tendon",
    "index",
    "middle",
    "ring",
    "pinky",
]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TetherIA – Aero Hand")
        self.geometry("900x620")
        self.minsize(860, 560)

        # runtime state
        self.hand: AeroHand | None = None
        self.tx_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.control_paused = False  # pause streaming during blocking ops
        self.tx_rate_hz = 40.0       # streaming rate for CTRL_POS
        self.slider_vars: list[tk.IntVar] = []
        self.port_var = tk.StringVar()
        self.baud_var = tk.IntVar(value=921600)

        self._build_ui()
        self._refresh_ports()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        # Port + refresh
        ttk.Label(top, text="Port:").pack(side=tk.LEFT)
        self.port_cmb = ttk.Combobox(top, textvariable=self.port_var, width=20, state="readonly")
        self.port_cmb.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(top, text="Refresh", command=self._refresh_ports).pack(side=tk.LEFT, padx=(0, 12))

        # Baud select
        ttk.Label(top, text="Baud:").pack(side=tk.LEFT)
        self.baud_cmb = ttk.Combobox(top, width=10, state="readonly",
                                     values=[str(b) for b in BAUDS], textvariable=self.baud_var)
        self.baud_cmb.set(str(self.baud_var.get()))
        self.baud_cmb.pack(side=tk.LEFT, padx=(4, 12))

        # Connect / Disconnect
        self.btn_connect = ttk.Button(top, text="Connect", command=self.on_connect)
        self.btn_connect.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_disc = ttk.Button(top, text="Disconnect", command=self.on_disconnect, state=tk.DISABLED)
        self.btn_disc.pack(side=tk.LEFT, padx=(0, 16))
        
        # Streaming rate
        ttk.Label(top, text="Rate (Hz):").pack(side=tk.LEFT)
        self.rate_spin = ttk.Spinbox(top, from_=1, to=200, width=6)
        self.rate_spin.delete(0, tk.END)
        self.rate_spin.insert(0, "40")
        self.rate_spin.pack(side=tk.LEFT, padx=(4, 0))

        # ---- Commands row
        cmd = ttk.Frame(self, padding=(10, 4))
        cmd.pack(side=tk.TOP, fill=tk.X)

        self.btn_homing = ttk.Button(cmd, text="Homing", command=self.on_homing, state=tk.DISABLED)
        self.btn_homing.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_setid = ttk.Button(cmd, text="Set ID", command=self.on_set_id, state=tk.DISABLED)
        self.btn_setid.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_trim = ttk.Button(cmd, text="Trim Servo", command=self.on_trim, state=tk.DISABLED)
        self.btn_trim.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_flash = ttk.Button(cmd, text="Upload Firmware", command=self.on_flash)
        self.btn_flash.pack(side=tk.LEFT, padx=(0, 10))

        # Zero All Button
        self.btn_zero = ttk.Button(cmd, text="Zero All", command=self.on_zero_all, state=tk.DISABLED)
        self.btn_zero.pack(side=tk.LEFT, padx=(0, 10))

        # GET buttons
        self.btn_get_pos  = ttk.Button(cmd, text="GET_POS",  command=self.on_get_pos,  state=tk.DISABLED)
        self.btn_get_vel  = ttk.Button(cmd, text="GET_VEL",  command=self.on_get_vel,  state=tk.DISABLED)
        self.btn_get_cur  = ttk.Button(cmd, text="GET_CURR", command=self.on_get_cur,  state=tk.DISABLED)
        self.btn_get_temp = ttk.Button(cmd, text="GET_TEMP", command=self.on_get_temp, state=tk.DISABLED)
        self.btn_get_pos.pack(side=tk.LEFT, padx=(20, 6))
        self.btn_get_vel.pack(side=tk.LEFT, padx=6)
        self.btn_get_cur.pack(side=tk.LEFT, padx=6)
        self.btn_get_temp.pack(side=tk.LEFT, padx=6)

        # ---- Sliders (7)
        grp = ttk.LabelFrame(self, text="Sliders (send CTRL_POS payload 0..65535)", padding=10)
        grp.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

        for i, name in enumerate(SLIDER_LABELS):
            row = ttk.Frame(grp)
            row.pack(fill=tk.X, pady=5)

            ttk.Label(row, text=f"{i} – {name}", width=18).pack(side=tk.LEFT)
            var = tk.IntVar(value=0)
            self.slider_vars.append(var)
            scale = tk.Scale(row, from_=0, to=65535, orient=tk.HORIZONTAL, length=600,
                             resolution=1, variable=var, showvalue=True)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
            val_lbl = ttk.Label(row, text="0", width=8)
            val_lbl.pack(side=tk.LEFT)
            var.trace_add("write", lambda *_a, v=var, lbl=val_lbl: lbl.config(text=str(v.get())))

        # ---- RX log
        rx = ttk.LabelFrame(self, text="RX Log", padding=10)
        rx.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.rx_text = tk.Text(rx, height=10)
        self.rx_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(rx, command=self.rx_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.rx_text.configure(yscrollcommand=sb.set)

        # ---- statusbar
        self.status_var = tk.StringVar(value="Disconnected")
        status_bar = ttk.Frame(self)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6))

        ttk.Label(status_bar, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(status_bar, text="Clear Log", command=self._clear_rx).pack(side=tk.RIGHT)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------- helpers -------------
    def log(self, s: str):
        self.rx_text.insert(tk.END, s + ("\n" if not s.endswith("\n") else ""))
        self.rx_text.see(tk.END)

    def set_status(self, s: str):
        self.status_var.set(s)

    def _refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        self.port_cmb["values"] = ports
        # auto-select the first port if none chosen
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        if not ports and not self.port_var.get():
            # plausible defaults
            if sys.platform.startswith("win"):
                self.port_var.set("COM12")
            else:
                self.port_var.set("/dev/ttyUSB0")

    # ------------- connect/disconnect -------------
    def on_connect(self):
        if self.hand is not None:
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Error", "Select a serial port.")
            return
        try:
            self.tx_rate_hz = float(self.rate_spin.get().strip())
            if self.tx_rate_hz <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Rate must be a positive number.")
            return

        baud = int(self.baud_var.get())
        try:
            self.hand = AeroHand(port, baudrate=baud)

            self.stop_event.clear()
            self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
            self.tx_thread.start()

            self.btn_connect.configure(state=tk.DISABLED)
            self.btn_disc.configure(state=tk.NORMAL)
            for b in (self.btn_zero,self.btn_homing, self.btn_setid, self.btn_trim,
                      self.btn_get_pos, self.btn_get_vel, self.btn_get_cur, self.btn_get_temp):
                b.configure(state=tk.NORMAL)

            self.set_status(f"Connected to {port} @ {baud}")
            self.log(f"[info] Connected {port} @ {baud}")
        except Exception as e:
            self.hand = None
            messagebox.showerror("Open failed", str(e))

    def on_disconnect(self):
        self._shutdown_serial()

    def _shutdown_serial(self):
        self.control_paused = True
        self.stop_event.set()
        if self.tx_thread and self.tx_thread.is_alive():
            try:
                self.tx_thread.join(timeout=0.5)
            except Exception:
                pass
        if self.hand:
            try:
                self.hand.close()
            except Exception:
                pass
        self.hand = None
        self.tx_thread = None
        self.control_paused = False

        self.btn_connect.configure(state=tk.NORMAL)
        self.btn_disc.configure(state=tk.DISABLED)
        for b in (self.btn_zero,self.btn_homing, self.btn_setid, self.btn_trim,
                  self.btn_get_pos, self.btn_get_vel, self.btn_get_cur, self.btn_get_temp):
            b.configure(state=tk.DISABLED)
        self.set_status("Disconnected")
        self.log("[info] Disconnected")

    # ------------- TX streaming (CTRL_POS) -------------
    def _tx_loop(self):
        period = 1.0 / max(1e-3, self.tx_rate_hz)
        next_t = time.perf_counter()
        while not self.stop_event.is_set():
            if self.hand is not None and not self.control_paused:
                payload = [int(v.get()) & 0xFFFF for v in self.slider_vars]
                try:
                    self.hand._send_data(CTRL_POS, payload)
                except Exception as e:
                    self.log(f"[TX error] {e}")
            # pacing
            next_t += period
            to_sleep = next_t - time.perf_counter()
            if to_sleep > 0:
                time.sleep(to_sleep)
            else:
                next_t = time.perf_counter()

    def on_homing(self):
        if not self.hand:
            return

        def worker():
            try:
                self.control_paused = True
                self.set_status("Homing in process… waiting for ACK")
                self.log("[TX] HOMING sent (0x01). Waiting for 16-byte ACK…")
                ok = self.hand.send_homing(timeout_s=100.0)
                if ok:
                    self.log("[ACK] HOMING complete.")
                    self.set_status("Homing complete")
            except Exception as e:
                self.log(f"[err] HOMING failed: {e}")
                self.set_status("Homing failed")
            finally:
                self.control_paused = False

        threading.Thread(target=worker, daemon=True).start()

    def on_set_id(self):
        if not self.hand:
            return
        new_id = simpledialog.askinteger("Set ID", "Enter new ID (0..253):", minvalue=0, maxvalue=253, parent=self)
        if new_id is None:
            return
        cur_lim = simpledialog.askinteger("Current Limit", "Enter current limit (0..1023):",
                                          minvalue=0, maxvalue=1023, initialvalue=1023, parent=self)
        if cur_lim is None:
            return

        def worker():
            try:
                self.control_paused = True
                self.set_status("Setting ID… waiting for ACK")
                self.log(f"[TX] SET_ID sent (id={new_id}, current={cur_lim})")
                ack = self.hand.set_id(new_id, cur_lim)  # dict with Old_id, New_id, Current_limit
                self.log(f"[ACK] SET_ID: old={ack['Old_id']} new={ack['New_id']} current_limit={ack['Current_limit']}")
                self.set_status("Set ID complete")
            except Exception as e:
                self.log(f"[err] SET_ID failed: {e}")
                self.set_status("Set ID failed")
            finally:
                self.control_paused = False

        threading.Thread(target=worker, daemon=True).start()

    def on_zero_all(self):
        if not self.hand:
            return

        def worker():
            try:
                self.control_paused = True
                for var in self.slider_vars:
                    var.set(0)
                joint_pos = list(self.hand.joint_lower_limits)  # 16 values
                self.hand.set_joint_positions(joint_pos)
                self.log("[TX] ZERO_ALL via CTRL_POS (joint lower limits)")
                self.set_status("Zeroed (lower limits sent; sliders reset)")
            except Exception as e:
                self.log(f"[err] ZERO_ALL: {e}")
                self.set_status("Zero All failed")
            finally:
                time.sleep(0.05)
                self.control_paused = False

        threading.Thread(target=worker, daemon=True).start()

    def on_trim(self):
        if not self.hand:
            return
        ch = simpledialog.askinteger("Trim Servo", "Servo ID / channel (0..14):",
                                     minvalue=0, maxvalue=14, parent=self)
        if ch is None:
            return
        deg = simpledialog.askinteger("Trim Servo", "Degrees (-360..360):",
                                      minvalue=-360, maxvalue=360, initialvalue=0, parent=self)
        if deg is None:
            return

        def worker():
            try:
                self.control_paused = True
                self.set_status("Trimming… waiting for ACK")
                self.log(f"[TX] TRIM sent (ch={ch}, deg={deg})")
                ack = self.hand.trim_servo(ch, deg)  # dict with Servo ID, Extend Count
                self.log(f"[ACK] TRIM: id={ack['Servo ID']} extend={ack['Extend Count']}")
                self.set_status("Trim complete")
            except Exception as e:
                self.log(f"[err] TRIM failed: {e}")
                self.set_status("Trim failed")
            finally:
                self.control_paused = False

        threading.Thread(target=worker, daemon=True).start()

    # ---- GET_* buttons (request + show parsed reply) ----
    def on_get_pos(self):
        if not self.hand:
            return
        try:
            vals = self.hand.get_motor_positions()
            self.log(f"[GET_POS] {list(vals)}")
        except Exception as e:
            self.log(f"[err] GET_POS: {e}")

    def on_get_vel(self):
        if not self.hand:
            return
        try:
            vals = self.hand.get_motor_speed()
            self.log(f"[GET_VEL] {list(vals)}")
        except Exception as e:
            self.log(f"[err] GET_VEL: {e}")

    def on_get_cur(self):
        if not self.hand:
            return
        try:
            vals = self.hand.get_motor_currents()
            self.log(f"[GET_CURR] {list(vals)}")
        except Exception as e:
            self.log(f"[err] GET_CURR: {e}")

    def on_get_temp(self):
        if not self.hand:
            return
        try:
            vals = self.hand.get_motor_temperatures()
            self.log(f"[GET_TEMP] {list(vals)}")
        except Exception as e:
            self.log(f"[err] GET_TEMP: {e}")

    # ---- Flashing (esptool) ----
    def on_flash(self):
        bin_path = filedialog.askopenfilename(
            parent=self, title="Select ESP32 firmware (.bin)",
            filetypes=[("BIN files", "*.bin"), ("All files", "*.*")]
        )
        if not bin_path:
            return
        # pick a port (use connected one if available)
        port = self.port_var.get().strip() or simpledialog.askstring("Port", "Enter serial port:", parent=self)
        if not port:
            return

        # Best effort chip guess; you can hardcode "esp32s3" if you want
        chip = "auto"

        # Decide offset: merged at 0x0, otherwise 0x10000
        name = os.path.basename(bin_path).lower()
        merged = ("merged" in name) or ("with_bootloader" in name) or name.endswith(".merged.bin")
        offset = "0x0" if merged else "0x10000"

        # Close serial if open
        if self.hand:
            self.log("[flash] Closing serial before flashing…")
            self.on_disconnect()

        def worker():
            cmd = [sys.executable, "-m", "esptool",
                   "--chip", chip, "-p", port, "-b", "921600",
                   "write-flash"] + ([] if merged else ["-z"]) + [offset, bin_path]
            self.log("> " + " ".join(cmd))
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.log(line.rstrip("\n"))
                rc = proc.wait()
                if rc == 0:
                    self.log("[flash] Flash complete.")
                    messagebox.showinfo("Success", "Firmware flashed successfully.")
                else:
                    self.log(f"[flash] esptool exited with code {rc}")
                    messagebox.showerror("Flash failed", f"esptool exited with code {rc}")
            except FileNotFoundError:
                self.log("[flash] esptool not found. Install with: pip install esptool")
                messagebox.showerror("Flash failed", "esptool not found. pip install esptool")
            except Exception as e:
                self.log(f"[flash] {e}")
                messagebox.showerror("Flash failed", str(e))

            # try to reconnect using current UI selections
            try:
                self.on_connect()
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    # ---- to clear RX window 
    def _clear_rx(self):
        """Clear the RX log text box."""
        try:
            self.rx_text.delete("1.0", tk.END)
        except Exception:
            pass
    # ---- teardown ----
    def _on_close(self):
        try:
            self._shutdown_serial()
        finally:
            self.destroy()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()