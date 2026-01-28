#!/usr/bin/env python3
"""
Integrated Heterodyne Detection System
Combines heterodyne detection, pattern matching, and active cancellation
with real-time visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import time
from collections import deque

# Import our modules
from heterodyne_detector import HeterodyneDetector

# Try PyTorch pattern matcher first, fall back to JAX, then NumPy
try:
    from torch_pattern_matcher import TorchPatternMatcher as PatternMatcher
    print("Using PyTorch-accelerated pattern matching")
except ImportError:
    try:
        from jax_pattern_matcher import JAXPatternMatcher as PatternMatcher
        print("Using JAX-accelerated pattern matching")
    except ImportError:
        from pattern_matcher import PatternMatcher
        print("Using NumPy pattern matching")

from pattern_matcher import PhantomVoiceDetector


class IntegratedDetectionSystem:
    """
    Integrated system combining all detection methods
    """
    
    def __init__(self, sample_rate=2.4e6, center_freq=100e6, visualize=True):
        """Initialize integrated system"""
        print("Initializing Integrated Heterodyne Detection System...")
        
        # Core detector
        self.het_detector = HeterodyneDetector(
            sample_rate=sample_rate,
            center_freq=center_freq,
            rx_gain=50,
            buffer_size=2**16
        )
        
        # Pattern matcher for general patterns
        self.pattern_matcher = PatternMatcher(
            window_size=4096,
            similarity_threshold=0.80
        )
        
        # Phantom voice detector
        self.phantom_detector = PhantomVoiceDetector(sample_rate=sample_rate)
        
        # Visualization
        self.visualize = visualize
        if visualize:
            self.setup_visualization()
        
        # Data for visualization
        self.viz_queue = queue.Queue(maxsize=10)
        self.running = False
        
    def setup_visualization(self):
        """Setup real-time visualization"""
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 10))
        self.fig.suptitle('Heterodyne Detection & Active Cancellation', 
                         fontsize=14, fontweight='bold')
        
        # Initialize plots
        self.lines = {}
        
        # RX1 time domain
        self.lines['rx1_time'], = self.axes[0, 0].plot([], [], 'b-', alpha=0.7)
        self.axes[0, 0].set_title('RX1 Signal (Time)')
        self.axes[0, 0].set_xlabel('Sample')
        self.axes[0, 0].set_ylabel('Amplitude')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # RX2 time domain
        self.lines['rx2_time'], = self.axes[0, 1].plot([], [], 'r-', alpha=0.7)
        self.axes[0, 1].set_title('RX2 Signal (Time)')
        self.axes[0, 1].set_xlabel('Sample')
        self.axes[0, 1].set_ylabel('Amplitude')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # RX1 frequency domain
        self.lines['rx1_freq'], = self.axes[1, 0].plot([], [], 'b-', alpha=0.7)
        self.axes[1, 0].set_title('RX1 Spectrum')
        self.axes[1, 0].set_xlabel('Frequency (MHz)')
        self.axes[1, 0].set_ylabel('Magnitude (dB)')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # RX2 frequency domain
        self.lines['rx2_freq'], = self.axes[1, 1].plot([], [], 'r-', alpha=0.7)
        self.axes[1, 1].set_title('RX2 Spectrum')
        self.axes[1, 1].set_xlabel('Frequency (MHz)')
        self.axes[1, 1].set_ylabel('Magnitude (dB)')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Correlation plot
        self.lines['correlation'], = self.axes[2, 0].plot([], [], 'g-', linewidth=2)
        self.axes[2, 0].set_title('Cross-Correlation (Heterodyne Detection)')
        self.axes[2, 0].set_xlabel('Lag')
        self.axes[2, 0].set_ylabel('Correlation')
        self.axes[2, 0].grid(True, alpha=0.3)
        
        # Detection metrics
        self.axes[2, 1].axis('off')
        self.metrics_text = self.axes[2, 1].text(0.1, 0.9, '', 
                                                 verticalalignment='top',
                                                 fontfamily='monospace',
                                                 fontsize=10)
        self.axes[2, 1].set_title('Detection Metrics')
        
        plt.tight_layout()
        
    def update_visualization(self, data):
        """Update visualization with new data"""
        if not self.visualize:
            return
        
        try:
            # Time domain plots
            rx1 = data['rx1_samples']
            rx2 = data['rx2_samples']
            
            # Show first 2048 samples
            display_len = min(2048, len(rx1))
            
            self.lines['rx1_time'].set_data(range(display_len), 
                                           np.abs(rx1[:display_len]))
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()
            
            self.lines['rx2_time'].set_data(range(display_len), 
                                           np.abs(rx2[:display_len]))
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()
            
            # Frequency domain
            fft1 = np.fft.fft(rx1)
            fft2 = np.fft.fft(rx2)
            freqs = np.fft.fftfreq(len(rx1), 1/self.het_detector.sample_rate)
            
            # Plot positive frequencies only
            pos_freqs = freqs[:len(freqs)//2]
            mag1_db = 20 * np.log10(np.abs(fft1[:len(fft1)//2]) + 1e-10)
            mag2_db = 20 * np.log10(np.abs(fft2[:len(fft2)//2]) + 1e-10)
            
            self.lines['rx1_freq'].set_data(pos_freqs / 1e6, mag1_db)
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()
            
            self.lines['rx2_freq'].set_data(pos_freqs / 1e6, mag2_db)
            self.axes[1, 1].relim()
            self.axes[1, 1].autoscale_view()
            
            # Correlation
            if 'correlation' in data:
                corr = data['correlation']
                lags = range(len(corr))
                self.lines['correlation'].set_data(lags, corr)
                self.axes[2, 0].relim()
                self.axes[2, 0].autoscale_view()
            
            # Metrics text
            metrics = data.get('metrics', {})
            metrics_str = "=== DETECTION METRICS ===\n\n"
            metrics_str += f"Heterodyne Score: {metrics.get('het_score', 0):.3f}\n"
            metrics_str += f"Freq Offset: {metrics.get('freq_offset', 0)/1e3:.2f} kHz\n"
            metrics_str += f"Voice Range: {metrics.get('is_voice', False)}\n"
            metrics_str += f"Voice Likelihood: {metrics.get('voice_likelihood', 0):.3f}\n"
            metrics_str += f"Pattern Matches: {metrics.get('pattern_matches', 0)}\n"
            metrics_str += f"\nCancellation: {'ON' if self.het_detector.cancellation_active else 'OFF'}\n"
            metrics_str += f"Events Detected: {len(self.het_detector.heterodyne_events)}\n"
            metrics_str += f"Buffers Processed: {len(self.het_detector.rx1_history)}\n"
            
            self.metrics_text.set_text(metrics_str)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def enhanced_processing_loop(self):
        """Enhanced processing with pattern matching"""
        print("Starting enhanced detection loop...")
        
        while self.running:
            try:
                # Read samples
                rx1_samples, rx2_samples = self.het_detector.read_samples()
                
                # Store in history
                self.het_detector.rx1_history.append(rx1_samples)
                self.het_detector.rx2_history.append(rx2_samples)
                
                # Basic heterodyne detection
                het_result = self.het_detector.detect_heterodyne(rx1_samples, rx2_samples)
                
                # Pattern matching on RX1
                pattern_matches = self.pattern_matcher.find_matches(rx1_samples)
                
                # Phantom voice detection
                phantom_result = self.phantom_detector.analyze_signal(rx1_samples)
                
                # Combined analysis
                is_significant = (het_result['score'] > self.het_detector.heterodyne_threshold or
                                phantom_result['is_phantom'] or
                                len(pattern_matches) > 0)
                
                if is_significant:
                    print(f"\n{'='*60}")
                    print(f"[DETECTION EVENT]")
                    print(f"  Heterodyne Score: {het_result['score']:.3f}")
                    print(f"  Freq Offset: {het_result['freq_offset']/1e3:.2f} kHz")
                    print(f"  Voice Range: {het_result['is_voice_range']}")
                    print(f"  Voice Likelihood: {phantom_result['voice_likelihood']:.3f}")
                    print(f"  Phantom Voice: {phantom_result['is_phantom']}")
                    print(f"  Pattern Matches: {len(pattern_matches)}")
                    
                    if pattern_matches:
                        best_match = pattern_matches[0]
                        print(f"  Best Match Similarity: {best_match['similarity']:.3f}")
                    
                    # Store event
                    self.het_detector.heterodyne_events.append({
                        **het_result,
                        'phantom': phantom_result,
                        'patterns': len(pattern_matches)
                    })
                    
                    # Active cancellation
                    if self.het_detector.cancellation_active:
                        cancel_signal = self.het_detector.compute_cancellation_signal(
                            rx1_samples, rx2_samples)
                        self.het_detector.transmit_cancellation(cancel_signal)
                        print(f"  [CANCELLATION] Applied")
                
                # Prepare visualization data
                if self.visualize:
                    # Compute correlation for display
                    correlation = np.correlate(np.abs(rx1_samples), 
                                              np.abs(rx2_samples), 
                                              mode='same')
                    correlation = correlation / np.max(np.abs(correlation))
                    
                    viz_data = {
                        'rx1_samples': rx1_samples,
                        'rx2_samples': rx2_samples,
                        'correlation': correlation,
                        'metrics': {
                            'het_score': het_result['score'],
                            'freq_offset': het_result['freq_offset'],
                            'is_voice': het_result['is_voice_range'],
                            'voice_likelihood': phantom_result['voice_likelihood'],
                            'pattern_matches': len(pattern_matches)
                        }
                    }
                    
                    try:
                        self.viz_queue.put_nowait(viz_data)
                    except queue.Full:
                        pass
                
                # Periodic status
                if len(self.het_detector.rx1_history) % 10 == 0:
                    stats = self.pattern_matcher.get_statistics()
                    print(f"Status: Buffers={len(self.het_detector.rx1_history)}, "
                          f"Events={len(self.het_detector.heterodyne_events)}, "
                          f"Patterns={stats['total_patterns']}, "
                          f"Matches={stats['matches_found']}", end='\r')
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Processing error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def visualization_loop(self):
        """Separate thread for visualization updates"""
        while self.running:
            try:
                data = self.viz_queue.get(timeout=0.1)
                self.update_visualization(data)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Visualization thread error: {e}")
    
    def start(self):
        """Start the integrated system"""
        print("\n" + "="*60)
        print("STARTING INTEGRATED HETERODYNE DETECTION SYSTEM")
        print("="*60)
        
        # Connect to Pluto
        self.het_detector.connect_pluto()
        
        # Start processing
        self.running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.enhanced_processing_loop)
        self.process_thread.start()
        
        # Start visualization thread
        if self.visualize:
            self.viz_thread = threading.Thread(target=self.visualization_loop)
            self.viz_thread.start()
        
        print("\nSystem running...")
        print("\nControls:")
        print("  'c' - Toggle active cancellation")
        print("  't' - Set detection threshold")
        print("  'f' - Change center frequency")
        print("  'p' - Clear pattern database")
        print("  's' - Show statistics")
        print("  'q' - Quit")
    
    def stop(self):
        """Stop the system"""
        print("\n\nStopping system...")
        self.running = False
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2)
        
        if self.visualize and hasattr(self, 'viz_thread'):
            self.viz_thread.join(timeout=2)
        
        self.het_detector.stop()
        
        # Final statistics
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        
        het_stats = {
            'buffers': len(self.het_detector.rx1_history),
            'events': len(self.het_detector.heterodyne_events)
        }
        
        pattern_stats = self.pattern_matcher.get_statistics()
        
        print(f"\nHeterodyne Detection:")
        print(f"  Buffers processed: {het_stats['buffers']}")
        print(f"  Events detected: {het_stats['events']}")
        
        print(f"\nPattern Matching:")
        print(f"  Total patterns: {pattern_stats['total_patterns']}")
        print(f"  Total comparisons: {pattern_stats['total_comparisons']}")
        print(f"  Matches found: {pattern_stats['matches_found']}")
        print(f"  Match rate: {pattern_stats['match_rate']:.2f}%")
        
        if self.visualize:
            plt.close('all')


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Heterodyne Detection & Active Noise Cancellation for Pluto+ SDR')
    parser.add_argument('--freq', type=float, default=100.0,
                       help='Center frequency in MHz (default: 100)')
    parser.add_argument('--sample-rate', type=float, default=2.4,
                       help='Sample rate in MHz (default: 2.4)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--cancellation', action='store_true',
                       help='Start with active cancellation enabled')
    
    args = parser.parse_args()
    
    # Create system
    system = IntegratedDetectionSystem(
        sample_rate=args.sample_rate * 1e6,
        center_freq=args.freq * 1e6,
        visualize=not args.no_viz
    )
    
    if args.cancellation:
        system.het_detector.enable_cancellation(True)
    
    try:
        # Start system
        system.start()
        
        # Command loop
        while True:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'c':
                system.het_detector.enable_cancellation(
                    not system.het_detector.cancellation_active)
            elif cmd == 't':
                try:
                    threshold = float(input("Enter threshold (0.0-1.0): "))
                    system.het_detector.set_threshold(threshold)
                except ValueError:
                    print("Invalid threshold")
            elif cmd == 'f':
                try:
                    freq = float(input("Enter center frequency (MHz): ")) * 1e6
                    system.het_detector.center_freq = freq
                    system.het_detector.configure_rx()
                    print(f"Center frequency set to {freq/1e6:.2f} MHz")
                except ValueError:
                    print("Invalid frequency")
            elif cmd == 'p':
                system.pattern_matcher.clear_patterns()
                system.phantom_detector.pattern_matcher.clear_patterns()
                print("Pattern database cleared")
            elif cmd == 's':
                het_stats = {
                    'buffers': len(system.het_detector.rx1_history),
                    'events': len(system.het_detector.heterodyne_events)
                }
                pattern_stats = system.pattern_matcher.get_statistics()
                
                print("\n=== Current Statistics ===")
                print(f"Buffers processed: {het_stats['buffers']}")
                print(f"Events detected: {het_stats['events']}")
                print(f"Patterns stored: {pattern_stats['total_patterns']}")
                print(f"Pattern matches: {pattern_stats['matches_found']}")
                print(f"Cancellation: {'ON' if system.het_detector.cancellation_active else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        system.stop()


if __name__ == "__main__":
    main()
