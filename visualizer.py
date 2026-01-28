#!/usr/bin/env python3
"""
Dash Plotly Visualization for Radar System - 3x3 Dashboard
Plots:
  Row 1: RX1 Time | RX2 Time | FFT Spectrum
  Row 2: MFCC Heatmap | Range-Doppler Map | PPI Polar
  Row 3: DOA Polar | Target Tracking | Track History
  
20 Hz refresh rate (50 ms) for 9 plots
"""

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
            "mfcc": np.zeros((13, 20)),
            "detection": {"score": 0.0, "freq_offset": 0.0},
            "geometry_info": {"doa": 0.0, "doa_power": np.zeros(37), "snr_improvement": 0.0},
            "range_doppler_map": np.zeros((128, 512)),
            "ppi": {"ppi_map": np.zeros((37, 256)), "angles_deg": np.linspace(-90, 90, 37)},
            "tracks": [],
        }
        self._lock = threading.Lock()

        self._setup_layout()
        self._setup_callbacks()
        self._thread = threading.Thread(target=self._run_server, daemon=True)

    def update(self, rx1, rx2, mfcc, detection, geometry_info,
               range_doppler_map, ppi, tracks):
        """Update visualizer with all processed radar data"""
        with self._lock:
            self._data["rx1"] = rx1
            self._data["rx2"] = rx2
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
                    "Radar System Dashboard - 3x3 Grid (20 Hz)",
                    style={"textAlign": "center", "color": "#00FFFF", "marginBottom": "20px"},
                ),
                
                # Row 1: Time Domain + FFT
                html.Div(
                    [
                        dcc.Graph(id="rx1_wave", style={"flex": "1"}),
                        dcc.Graph(id="rx2_wave", style={"flex": "1"}),
                        dcc.Graph(id="fft_spectrum", style={"flex": "1"}),
                    ],
                    style={
                        "display": "flex",
                        "gap": "10px",
                        "marginBottom": "10px",
                    },
                ),
                
                # Row 2: MFCC + Range-Doppler + PPI
                html.Div(
                    [
                        dcc.Graph(id="mfcc_heatmap", style={"flex": "1"}),
                        dcc.Graph(id="range_doppler", style={"flex": "1"}),
                        dcc.Graph(id="ppi_polar", style={"flex": "1"}),
                    ],
                    style={
                        "display": "flex",
                        "gap": "10px",
                        "marginBottom": "10px",
                    },
                ),
                
                # Row 3: DOA Polar + Tracking + History
                html.Div(
                    [
                        dcc.Graph(id="doa_polar", style={"flex": "1"}),
                        dcc.Graph(id="target_tracking", style={"flex": "1"}),
                        dcc.Graph(id="track_history", style={"flex": "1"}),
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
        
        # ========== Row 1: Time Domain + FFT ==========
        
        @self.app.callback(
            Output("rx1_wave", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_rx1_time(_n):
            with self._lock:
                data = self._data["rx1"]
            y = np.abs(data[:1024])
            # Decimate for display performance
            y = y[::10]
            fig = go.Figure(
                data=[go.Scatter(y=y, mode="lines", line=dict(color="#00FFFF", width=1))]
            )
            fig.update_layout(
                title="RX1 Time Domain",
                template="plotly_dark",
                xaxis_title="Sample (decimated)",
                yaxis_title="Amplitude",
                height=350,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("rx2_wave", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_rx2_time(_n):
            with self._lock:
                data = self._data["rx2"]
            y = np.abs(data[:1024])
            y = y[::10]
            fig = go.Figure(
                data=[go.Scatter(y=y, mode="lines", line=dict(color="#FF00FF", width=1))]
            )
            fig.update_layout(
                title="RX2 Time Domain",
                template="plotly_dark",
                xaxis_title="Sample (decimated)",
                yaxis_title="Amplitude",
                height=350,
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
            fig = go.Figure(
                data=[go.Scatter(y=mag_db, mode="lines", fill="tozeroy", 
                                line=dict(color="#00FF00"))]
            )
            fig.update_layout(
                title="RX1 Spectrum (dB)",
                template="plotly_dark",
                xaxis_title="Bin",
                yaxis_title="Power (dB)",
                height=350,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        # ========== Row 2: MFCC + Range-Doppler + PPI ==========
        
        @self.app.callback(
            Output("mfcc_heatmap", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_mfcc(_n):
            with self._lock:
                mfcc = self._data["mfcc"]
            fig = go.Figure(data=go.Heatmap(
                z=mfcc,
                colorscale="Viridis",
                colorbar=dict(thickness=15, len=0.7)
            ))
            fig.update_layout(
                title="MFCC Features",
                template="plotly_dark",
                xaxis_title="Time Frame",
                yaxis_title="MFCC Coeff",
                height=350,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("range_doppler", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_range_doppler(_n):
            with self._lock:
                rd_map = self._data["range_doppler_map"]
            
            # Handle empty map
            if rd_map.size == 0:
                rd_map = np.zeros((128, 512))
            
            fig = go.Figure(data=go.Heatmap(
                z=rd_map,
                colorscale="Jet",
                colorbar=dict(title="dB", thickness=15, len=0.7)
            ))
            fig.update_layout(
                title="Range-Doppler Map",
                template="plotly_dark",
                xaxis_title="Range Bin",
                yaxis_title="Doppler Bin",
                height=350,
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
            angles_deg = ppi_data.get("angles_deg", np.linspace(-90, 90, 37))
            
            # Create polar plot data
            # Convert Cartesian PPI map to polar coordinates for display
            fig = go.Figure(data=go.Heatmap(
                z=ppi_map,
                colorscale="Viridis",
                colorbar=dict(title="dB", thickness=15, len=0.7)
            ))
            fig.update_layout(
                title="PPI - Polar Display",
                template="plotly_dark",
                xaxis_title="Range Bin",
                yaxis_title="Angle (°)",
                height=350,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        # ========== Row 3: DOA + Tracking + History ==========
        
        @self.app.callback(
            Output("doa_polar", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_doa(_n):
            with self._lock:
                geo_info = self._data["geometry_info"]
            
            doa_power = geo_info.get("doa_power", np.zeros(37))
            angles_deg = np.linspace(-90, 90, 37)
            
            fig = go.Figure(data=go.Scatterpolar(
                r=doa_power,
                theta=angles_deg,
                fill="toself",
                name="DOA Power",
                line=dict(color="#00FFFF"),
            ))
            fig.update_layout(
                title="Direction of Arrival (DOA)",
                template="plotly_dark",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, np.max(doa_power) + 1e-10]),
                ),
                height=350,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            return fig

        @self.app.callback(
            Output("target_tracking", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_tracking(_n):
            with self._lock:
                tracks = self._data["tracks"]
            
            fig = go.Figure()
            
            # Plot each track
            for track in tracks:
                if isinstance(track, dict):
                    x = track.get("x", 0)
                    y = track.get("y", 0)
                    vx = track.get("vx", 0)
                    vy = track.get("vy", 0)
                    track_id = track.get("track_id", 0)
                    
                    # Plot position
                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers",
                        marker=dict(size=10, color="#00FF00"),
                        name=f"Track {track_id}",
                    ))
                    
                    # Plot velocity vector
                    if vx != 0 or vy != 0:
                        fig.add_trace(go.Scatter(
                            x=[x, x + vx],
                            y=[y, y + vy],
                            mode="lines",
                            line=dict(color="#00FF00", width=2),
                            showlegend=False,
                        ))
            
            fig.update_layout(
                title="Target Tracking (Cartesian)",
                template="plotly_dark",
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                height=350,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            
            return fig

        @self.app.callback(
            Output("track_history", "figure"), Input("dash-interval", "n_intervals")
        )
        def update_track_history(_n):
            with self._lock:
                tracks = self._data["tracks"]
            
            fig = go.Figure()
            
            # Plot trajectory history for each track
            colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
            for i, track in enumerate(tracks):
                if isinstance(track, dict):
                    history = track.get("history")
                    if history is not None and len(history) > 0:
                        history = np.array(history)
                        color = colors[i % len(colors)]
                        fig.add_trace(go.Scatter(
                            x=history[:, 0],
                            y=history[:, 1],
                            mode="lines+markers",
                            name=f"Track {track.get('track_id', i)}",
                            line=dict(color=color),
                            marker=dict(size=4),
                        ))
            
            fig.update_layout(
                title="Track History (Trajectories)",
                template="plotly_dark",
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                height=350,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            
            return fig

    def _run_server(self):
        """Run the Dash server"""
        self.app.run(host=self.host, port=self.port, debug=False)

    def start(self):
        """Start the visualization server in background thread"""
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        """Stop the visualizer (placeholder for cleanup)"""
        # Dash/Flask doesn't have a simple 'stop' for the development server
        # when running in a thread, but we can signal cleanup if needed.
        print("Stopping Visualizer...")
