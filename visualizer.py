#!/usr/bin/env python3
"""
Dash Plotly Visualization for Radar System - 3x3 Dashboard (Async)
Plots:
  Row 1: RX1 Time | RX2 Time | FFT Spectrum
  Row 2: MFCC Heatmap | Range-Doppler Map | PPI Polar
  Row 3: DOA Polar | Target Tracking | Track History

20 Hz refresh rate (50 ms) for 9 plots
Non-blocking async updates via asyncio.Queue
"""

import asyncio
import threading
import time
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go


class VisualizerDash:
    def __init__(
        self, refresh_rate_hz=20, geometry=None, host="127.0.0.1", port=8050, **kwargs
    ):
        self.refresh_rate_hz = refresh_rate_hz
        self.geometry = geometry
        self.app = dash.Dash(__name__)
        self.host = host
        self.port = port

        # Latest data storage (expanded for new plots)
        self._data = {
            "rx1": np.zeros(1024),
            "rx2": np.zeros(1024),
            "tx1": np.zeros(1024),
            "tx2": np.zeros(1024),
            "mfcc": np.zeros((13, 20)),
            "detection": {"score": 0.0, "freq_offset": 0.0},
            "geometry_info": {"doa": 0.0, "doa_power": np.zeros(37), "snr_improvement": 0.0},
            "range_doppler_map": np.zeros((128, 512)),
            "ppi": {"ppi_map": np.zeros((37, 256)), "angles_deg": np.linspace(-90, 90, 37)},
            "tracks": [],
        }

        # Async-friendly: Use asyncio.Lock instead of threading.Lock
        # Data access is through synchronized queue pattern
        self._lock = threading.Lock()  # Keep for Dash callbacks (they run in Flask threads)
        self._update_queue = None  # Will be set when async loop is available

        self._setup_layout()
        self._setup_callbacks()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._running = False

    async def update(self, rx1, rx2, mfcc, detection, geometry_info,
                     range_doppler_map, ppi, tracks, tx1=None, tx2=None):
        """
        Async update - non-blocking visualization refresh.
        Uses asyncio.sleep(0) to yield control immediately.
        """
        # Update data atomically (fast operation, safe to lock briefly)
        with self._lock:
            self._data["rx1"] = rx1
            self._data["rx2"] = rx2
            if tx1 is not None: self._data["tx1"] = tx1
            if tx2 is not None: self._data["tx2"] = tx2
            self._data["mfcc"] = mfcc
            self._data["detection"] = detection
            self._data["geometry_info"] = geometry_info
            self._data["range_doppler_map"] = range_doppler_map
            self._data["ppi"] = ppi
            self._data["tracks"] = tracks

        # Yield control immediately (don't block event loop)
        await asyncio.sleep(0)

    def update_sync(self, rx1, rx2, mfcc, detection, geometry_info,
                    range_doppler_map, ppi, tracks, tx1=None, tx2=None):
        """Synchronous fallback for compatibility"""
        with self._lock:
            self._data["rx1"] = rx1
            self._data["rx2"] = rx2
            if tx1 is not None: self._data["tx1"] = tx1
            if tx2 is not None: self._data["tx2"] = tx2
            self._data["mfcc"] = mfcc
            self._data["detection"] = detection
            self._data["geometry_info"] = geometry_info
            self._data["range_doppler_map"] = range_doppler_map
            self._data["ppi"] = ppi
            self._data["tracks"] = tracks

    def _setup_layout(self):
        """Setup 3x3 grid dashboard layout"""
        interval_ms = int(1000 / self.refresh_rate_hz)
        
        self.app.layout = html.Div(
            [
                html.H2(
                    "Radar System Dashboard - Torch-First 2TX2RX",
                    style={"textAlign": "center", "color": "#00FFFF", "marginBottom": "20px"},
                ),
                
                # Row 1: RX Wave (Combined) | TX1 Wave | TX2 Wave
                html.Div(
                    [
                        dcc.Graph(id="rx_wave", style={"flex": "1"}),
                        dcc.Graph(id="tx1_wave", style={"flex": "1"}),
                        dcc.Graph(id="tx2_wave", style={"flex": "1"}),
                    ],
                    style={
                        "display": "flex",
                        "gap": "10px",
                        "marginBottom": "10px",
                    },
                ),
                
                # Row 2: MFCC | Range-Doppler | FFT Spectrum
                html.Div(
                    [
                        dcc.Graph(id="mfcc_heatmap", style={"flex": "1"}),
                        dcc.Graph(id="range_doppler", style={"flex": "1"}),
                        dcc.Graph(id="fft_spectrum", style={"flex": "1"}),
                    ],
                    style={
                        "display": "flex",
                        "gap": "10px",
                        "marginBottom": "10px",
                    },
                ),
                
                # Row 3: DOA Polar | Target Tracking | PPI Polar
                html.Div(
                    [
                        dcc.Graph(id="doa_polar", style={"flex": "1"}),
                        dcc.Graph(id="target_tracking", style={"flex": "1"}),
                        dcc.Graph(id="ppi_polar", style={"flex": "1"}),
                    ],
                    style={
                        "display": "flex",
                        "gap": "10px",
                    },
                ),
                
                dcc.Interval(id="dash-interval", interval=interval_ms, n_intervals=0),
            ],
            style={
                "width": "100%",
                "height": "100%",
                "padding": "10px",
                "backgroundColor": "#111111",
            },
        )

    def _setup_callbacks(self):
        """Setup all 9 plot update callbacks"""
        
        # ========== Row 1: RX (Combined) | TX1 | TX2 ==========
        
        @self.app.callback(
            Output("rx_wave", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_rx_wave(_n):
            with self._lock:
                rx1 = self._data["rx1"]
                rx2 = self._data["rx2"]
            y1 = np.abs(rx1[:1024])[::10]
            y2 = np.abs(rx2[:1024])[::10]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y1, name="RX1", line=dict(color="#00FFFF", width=1)))
            fig.add_trace(go.Scatter(y=y2, name="RX2", line=dict(color="#FF00FF", width=1)))
            fig.update_layout(
                title="RX Time Domain",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("tx1_wave", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_tx1_wave(_n):
            with self._lock:
                data = self._data["tx1"]
            y = np.abs(data[:1024])[::10]
            fig = go.Figure(data=[go.Scatter(y=y, line=dict(color="#00FF00", width=1))])
            fig.update_layout(
                title="TX1 Cancellation Signal",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("tx2_wave", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_tx2_wave(_n):
            with self._lock:
                data = self._data["tx2"]
            y = np.abs(data[:1024])[::10]
            fig = go.Figure(data=[go.Scatter(y=y, line=dict(color="#FFFF00", width=1))])
            fig.update_layout(
                title="TX2 Cancellation Signal",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        # ========== Row 2: MFCC | RD | FFT ==========
        
        @self.app.callback(
            Output("mfcc_heatmap", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_mfcc(_n):
            with self._lock:
                mfcc = self._data["mfcc"]
            fig = go.Figure(data=go.Heatmap(z=mfcc, colorscale="Viridis"))
            fig.update_layout(
                title="MFCC Features",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("range_doppler", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_range_doppler(_n):
            with self._lock:
                rd_map = self._data["range_doppler_map"]
            fig = go.Figure(data=go.Heatmap(z=rd_map, colorscale="Jet"))
            fig.update_layout(
                title="Range-Doppler Map",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("fft_spectrum", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_fft(_n):
            with self._lock:
                data = self._data["rx1"]
            arr = data[:1024]
            mag = np.abs(np.fft.fft(np.asarray(arr, dtype=np.complex64)))[:512]
            mag_db = 20 * np.log10(mag + 1e-10)
            fig = go.Figure(data=[go.Scatter(y=mag_db, fill="tozeroy", line=dict(color="#00FF00"))])
            fig.update_layout(
                title="Spectral Power (dB)",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        # ========== Row 3: DOA | Tracking | PPI ==========
        
        @self.app.callback(
            Output("doa_polar", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_doa(_n):
            with self._lock:
                geo_info = self._data["geometry_info"]
            doa_power = geo_info.get("doa_power", np.zeros(37))
            angles_deg = np.linspace(-90, 90, 37)
            fig = go.Figure(data=go.Scatterpolar(r=doa_power, theta=angles_deg, fill="toself"))
            fig.update_layout(
                title="DOA Power Distribution",
                template="plotly_dark",
                polar=dict(radialaxis=dict(visible=False)),
                height=300,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("target_tracking", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_tracking(_n):
            with self._lock:
                tracks = self._data["tracks"]
            fig = go.Figure()
            for track in tracks:
                if isinstance(track, dict):
                    fig.add_trace(go.Scatter(x=[track.get("x", 0)], y=[track.get("y", 0)],
                                           mode="markers+text", name=f"T{track.get('track_id')}"))
            fig.update_layout(
                title="Target Tracking",
                template="plotly_dark",
                xaxis=dict(range=[-10, 10]),
                yaxis=dict(range=[0, 20]),
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("ppi_polar", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_ppi(_n):
            with self._lock:
                ppi_data = self._data["ppi"]
            ppi_map = ppi_data.get("ppi_map", np.zeros((37, 256)))
            fig = go.Figure(data=go.Heatmap(z=ppi_map, colorscale="Viridis"))
            fig.update_layout(
                title="PPI Polar Display",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

    def _run_server(self):
        """Run the Dash server with auto port finding"""
        import socket

        # Try to find available port
        port = self.port
        max_attempts = 10

        for attempt in range(max_attempts):
            try:
                # Test if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, port))
                    # Port is available
                    break
            except OSError:
                # Port in use, try next
                port += 1

        if port != self.port:
            print(f"⚠️  Port {self.port} in use, using {port} instead")
            self.port = port

        # Suppress Flask startup messages for cleaner output
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self.app.run(host=self.host, port=port, debug=False)

    def start(self):
        """Start the visualization server in background thread"""
        if not self._thread.is_alive():
            self._running = True
            self._thread.start()
            # Give server time to start and find port
            time.sleep(0.5)
            print(f"📊 Dashboard: http://{self.host}:{self.port}")

    async def start_async(self):
        """Async start - returns immediately after spawning server thread"""
        self.start()
        await asyncio.sleep(0.1)  # Let server initialize

    def stop(self):
        """Stop the visualizer (placeholder for cleanup)"""
        self._running = False
        print("Stopping Visualizer...")

    async def stop_async(self):
        """Async stop - non-blocking shutdown"""
        self._running = False
        await asyncio.sleep(0)
