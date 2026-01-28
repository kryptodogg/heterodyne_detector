# Product Guidelines: Heterodyne Detector with Spatial Noise Cancellation

## 1. Communication & Prose Style
- **Accessible & Educational:** Complex RF and signal processing concepts should be explained simply to remain accessible to a broader SDR and enthusiast audience.
- **Operational & Tactical:** Documentation and communication should be concise and action-oriented, prioritizing "how-to" guides and clear deployment instructions.

## 2. Development & "Vibe Coding" Principles
- **Modular Constraint:** Each module should strictly be between 250-500 lines to ensure the code remains easily manageable and fits within LLM context windows.
- **Self-Documenting Intent:** Code comments should focus on "Why" a decision was made rather than just "What" the code is doing, effectively guiding both human developers and LLMs.
- **Functional Purity:** Favor pure functions and immutable data structures to minimize side effects and simplify debugging.
- **Async/Non-blocking:** All processing should be non-blocking and utilize async/await patterns. As a GPU-centric application, responsiveness should rival high-end desktop/mobile operating systems (iOS/macOS).

## 3. User Experience (UX) & Dashboard
- **Real-Time Fluidity:** The dashboard must prioritize a 60fps refresh rate to ensure smooth data visualization and transitions.
- **Organized Density:** High information density is encouraged but must be well-organized. Use scrollable containers for large data sets like spectrograms or radar/sonar imaging.
- **Distributed Control:** The desktop application serves as the primary data processor and display, but it should be architected to allow remote control and data monitoring from other devices.

## 4. Edge Ecosystem & Integration
- **Protocol Standardization:** Enforce a standard data schema (JSON/Protobuf) for all incoming data from the edge ecosystem (ESP32, Raspberry Pi, Mobile, etc.).
- **Zero-Config Discovery:** Utilize mDNS or similar protocols for automatic discovery and pairing of edge sensors with the main server.
- **Network Resilience:** The system must gracefully handle edge device disconnections and reconnections without impacting the overall processing pipeline.

## 5. Diagnostics & Error Handling
- **Informative Notifications:** Provide clear, high-level status updates for common operational issues (e.g., SDR connection status).
- **Deep Traceability:** Maintain verbose logging of internal RF and GPU events to support expert-level debugging and performance analysis.
