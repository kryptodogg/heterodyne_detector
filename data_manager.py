#!/usr/bin/env python3
"""
HDF5 Library Management for Pattern Storage
Date > Session > Band Organization
User-controlled library paths and creation/loading
"""

import h5py
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from config import LIBRARY_CONFIG, DATASET_CONFIG, PREDEFINED_LABELS
import torch


class HDF5LibraryManager:
    """
    HDF5 Pattern Library Management
    Organized by Date > Session > Band hierarchy
    User-controlled paths and library operations
    """

    def __init__(self, base_path: Optional[str] = None):
        """Initialize library manager with specified path"""
        self.base_path = Path(base_path) if base_path else LIBRARY_CONFIG["base_path"]
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.current_library_path = None
        self.current_library = None
        self.current_session = None
        self.current_band = None

        # Statistics
        self.stats = {
            "total_patterns": 0,
            "total_size_mb": 0.0,
            "band_breakdown": {},
            "last_save": None,
        }

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_library(
        self, library_name: str, session_id: Optional[str] = None
    ) -> str:
        """Create new HDF5 library with proper hierarchy"""
        if session_id is None:
            session_id = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        today = datetime.now().strftime("%Y-%m-%d")

        # Library path: base/YYYY-MM-DD/Session_ID/library_name.h5
        session_path = self.base_path / today / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        library_path = session_path / f"{library_name}.h5"

        # Create HDF5 file with groups
        with h5py.File(library_path, "w") as f:
            # Create metadata group
            metadata_group = f.create_group("metadata")
            metadata_group.attrs["created"] = datetime.now().isoformat()
            metadata_group.attrs["session_id"] = session_id
            metadata_group.attrs["library_name"] = library_name
            metadata_group.attrs["version"] = "1.0"

            # Store config snapshot
            config_group = f.create_group("config")
            self._save_config_snapshot(config_group)

        self.current_library_path = library_path
        self.current_session = session_id

        return str(library_path)

    def load_library(self, library_path: str) -> bool:
        """Load existing HDF5 library"""
        path = Path(library_path)

        if not path.exists():
            return False

        try:
            self.current_library = h5py.File(path, "a")  # Append mode
            self.current_library_path = path

            # Load metadata
            if "metadata" in self.current_library:
                metadata = self.current_library["metadata"]
                self.current_session = metadata.attrs.get("session_id", "unknown")
            else:
                self.current_session = "unknown"

            # Update statistics
            self._update_library_stats()

            return True

        except Exception as e:
            print(f"Error loading library {library_path}: {e}")
            return False

    def save_pattern(
        self,
        mfcc_features: np.ndarray,
        metadata: Dict[str, Any],
        band: str = "2400MHz",
        raw_iq: Optional[np.ndarray] = None,
        stft_features: Optional[np.ndarray] = None,
        label: str = "unlabeled",
    ) -> str:
        """Save pattern to current library"""
        if self.current_library is None:
            raise ValueError(
                "No library loaded. Call create_library() or load_library() first."
            )

        # Create or get band group
        if band not in self.current_library:
            band_group = self.current_library.create_group(band)
            band_group.attrs["created"] = datetime.now().isoformat()
            band_group.attrs["band_center_freq"] = (
                ISM_BANDS[band]["center_freq"] if band in ISM_BANDS else 0
            )
        else:
            band_group = self.current_library[band]

        # Generate unique pattern ID
        pattern_id = f"pattern_{int(time.time() * 1000000)}_{len(band_group)}"

        # Create dataset for MFCC features
        mfcc_dataset = band_group.create_dataset(
            f"{pattern_id}/mfcc", data=mfcc_features, **DATASET_CONFIG["mfcc_features"]
        )
        mfcc_dataset.attrs["label"] = label
        mfcc_dataset.attrs["created"] = datetime.now().isoformat()

        # Save raw IQ data if provided
        if raw_iq is not None:
            iq_dataset = band_group.create_dataset(
                f"{pattern_id}/raw_iq", data=raw_iq, **DATASET_CONFIG["raw_iq"]
            )

        # Save STFT features if provided
        if stft_features is not None:
            stft_dataset = band_group.create_dataset(
                f"{pattern_id}/stft",
                data=stft_features,
                **DATASET_CONFIG["stft_features"],
            )

        # Save metadata
        pattern_group = band_group[f"{pattern_id}"]
        metadata_group = pattern_group.create_group("metadata")

        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                metadata_group.attrs[key] = value
            elif isinstance(value, (list, np.ndarray)):
                metadata_group.attrs[key] = json.dumps(value)
            else:
                metadata_group.attrs[key] = str(value)

        # Update statistics
        self.stats["total_patterns"] += 1
        if label not in self.stats["band_breakdown"]:
            self.stats["band_breakdown"][label] = 0
        self.stats["band_breakdown"][label] += 1

        return pattern_id

    def load_pattern(
        self, pattern_id: str, band: str = "2400MHz"
    ) -> Optional[Dict[str, Any]]:
        """Load specific pattern from library"""
        if self.current_library is None:
            return None

        if band not in self.current_library:
            return None

        band_group = self.current_library[band]
        if pattern_id not in band_group:
            return None

        pattern_group = band_group[pattern_id]

        # Load all data
        result = {}

        # Load MFCC features
        if "mfcc" in pattern_group:
            result["mfcc"] = pattern_group["mfcc"][:]

        # Load raw IQ
        if "raw_iq" in pattern_group:
            result["raw_iq"] = pattern_group["raw_iq"][:]

        # Load STFT
        if "stft" in pattern_group:
            result["stft"] = pattern_group["stft"][:]

        # Load metadata
        if "metadata" in pattern_group:
            metadata_group = pattern_group["metadata"]
            result["metadata"] = dict(metadata_group.attrs)

        return result

    def list_libraries(self, directory: Optional[str] = None) -> List[Dict[str, str]]:
        """List all available HDF5 libraries"""
        search_path = Path(directory) if directory else self.base_path

        libraries = []

        # Search recursively for .h5 files
        for h5_file in search_path.rglob("*.h5"):
            relative_path = h5_file.relative_to(search_path)

            # Extract date and session from path
            path_parts = relative_path.parts
            if len(path_parts) >= 2:
                date_part = path_parts[0]  # YYYY-MM-DD
                session_part = path_parts[1]  # Session_XXX
                library_name = h5_file.stem

                libraries.append(
                    {
                        "path": str(h5_file),
                        "name": library_name,
                        "date": date_part,
                        "session": session_part,
                        "full_path": relative_path,
                    }
                )

        return sorted(libraries, key=lambda x: (x["date"], x["session"], x["name"]))

    def get_storage_info(self) -> Dict[str, Any]:
        """Get comprehensive storage information"""
        if self.current_library is None:
            return {"status": "no_library_loaded"}

        # Calculate file size
        if self.current_library_path:
            file_size = Path(self.current_library_path).stat().st_size / (
                1024 * 1024
            )  # MB
        else:
            file_size = 0

        # Get band breakdown
        band_info = {}
        for band_name in self.current_library.keys():
            if isinstance(self.current_library[band_name], h5py.Group):
                patterns = [
                    k for k in self.current_library[band_name].keys() if "pattern_" in k
                ]
                band_info[band_name] = {
                    "patterns": len(patterns),
                    "labels": self._get_label_breakdown(band_name),
                }

        return {
            "total_patterns": self.stats["total_patterns"],
            "file_size_mb": file_size,
            "file_path": str(self.current_library_path),
            "session_id": self.current_session,
            "band_breakdown": band_info,
            "labels_breakdown": self.stats["band_breakdown"],
            "last_save": self.stats.get("last_save", None),
        }

    def manual_cleanup(self, criteria: Dict[str, Any]) -> int:
        """Manual cleanup based on user criteria"""
        if self.current_library is None:
            return 0

        deleted_count = 0

        for band_name in list(self.current_library.keys()):
            if not isinstance(self.current_library[band_name], h5py.Group):
                continue

            band_group = self.current_library[band_name]
            patterns_to_delete = []

            # Find patterns matching criteria
            for pattern_name in list(band_group.keys()):
                if not pattern_name.startswith("pattern_"):
                    continue

                pattern_group = band_group[pattern_name]

                # Check criteria
                if self._matches_cleanup_criteria(pattern_group, criteria):
                    patterns_to_delete.append(pattern_name)

            # Delete patterns
            for pattern_name in patterns_to_delete:
                del band_group[pattern_name]
                deleted_count += 1

        # Update statistics
        self._update_library_stats()

        return deleted_count

    def export_patterns(self, export_path: str, categories: Optional[List[str]] = None):
        """Export patterns to external file"""
        if self.current_library is None:
            return False

        export_data = {}

        for band_name in self.current_library.keys():
            if not isinstance(self.current_library[band_name], h5py.Group):
                continue

            band_group = self.current_library[band_name]
            band_data = []

            for pattern_name in band_group.keys():
                if not pattern_name.startswith("pattern_"):
                    continue

                pattern = self.load_pattern(pattern_name, band_name)
                if pattern:
                    # Filter by category if specified
                    if categories:
                        label = pattern.get("metadata", {}).get("label", "unlabeled")
                        if label not in categories:
                            continue

                    band_data.append(pattern)

            if band_data:
                export_data[band_name] = band_data

        # Save to file (JSON format for compatibility)
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        return True

    def close_library(self):
        """Close current library"""
        if self.current_library:
            self.current_library.close()
            self.current_library = None
            self.current_library_path = None

    def _save_config_snapshot(self, config_group: h5py.Group):
        """Save current configuration to HDF5"""
        import config

        config_snapshot = {
            "ism_bands": config.ISM_BANDS,
            "directional_probing": config.DIRECTIONAL_PROBING,
            "mfcc_config": config.MFCC_CONFIG,
            "gpu_config": config.GPU_CONFIG,
            "pattern_matching": config.PATTERN_MATCHING,
        }

        config_group.attrs["config_snapshot"] = json.dumps(config_snapshot)

    def _matches_cleanup_criteria(
        self, pattern_group: h5py.Group, criteria: Dict[str, Any]
    ) -> bool:
        """Check if pattern matches cleanup criteria"""
        metadata = dict(pattern_group.get("metadata", {}).attrs)

        # Check by label
        if "label" in criteria:
            if metadata.get("label") != criteria["label"]:
                return False

        # Check by age (days)
        if "older_than_days" in criteria:
            created_str = metadata.get("created")
            if created_str:
                created_date = datetime.fromisoformat(created_str)
                age_days = (datetime.now() - created_date).days
                if age_days <= criteria["older_than_days"]:
                    return False

        # Check by score
        if "score_less_than" in criteria:
            score = metadata.get("score", 1.0)
            if score >= criteria["score_less_than"]:
                return False

        return True

    def _get_label_breakdown(self, band_name: str) -> Dict[str, int]:
        """Get label breakdown for specific band"""
        if band_name not in self.current_library:
            return {}

        band_group = self.current_library[band_name]
        label_counts = {}

        for pattern_name in band_group.keys():
            if not pattern_name.startswith("pattern_"):
                continue

            pattern_group = band_group[pattern_name]
            metadata = dict(pattern_group.get("metadata", {}).attrs)
            label = metadata.get("label", "unlabeled")

            label_counts[label] = label_counts.get(label, 0) + 1

        return label_counts

    def _update_library_stats(self):
        """Update library statistics"""
        if self.current_library is None:
            return

        total_patterns = 0
        label_breakdown = {}

        for band_name in self.current_library.keys():
            if not isinstance(self.current_library[band_name], h5py.Group):
                continue

            band_labels = self._get_label_breakdown(band_name)
            for label, count in band_labels.items():
                label_breakdown[label] = label_breakdown.get(label, 0) + count
                total_patterns += count

        self.stats.update(
            {
                "total_patterns": total_patterns,
                "band_breakdown": label_breakdown,
                "last_save": datetime.now().isoformat(),
            }
        )


class GhostAudioLabeler:
    """
    Interactive labeling for ghost audio patterns
    Supports predefined and custom categories
    """

    def __init__(self, library_manager: HDF5LibraryManager):
        self.library_manager = library_manager
        self.current_pattern = None
        self.label_history = []

    def add_custom_category(self, category_name: str):
        """Add new custom label category"""
        if category_name not in PREDEFINED_LABELS:
            PREDEFINED_LABELS.append(category_name)

    def label_current_pattern(
        self,
        pattern_id: str,
        band: str,
        label: str = "unlabeled",
        custom_label: Optional[str] = None,
    ) -> bool:
        """Apply label to current pattern"""
        actual_label = custom_label if custom_label else label

        # Validate label
        valid_labels = PREDEFINED_LABELS + self._get_custom_labels()
        if actual_label not in valid_labels:
            print(f"Invalid label: {actual_label}")
            return False

        # Update pattern in HDF5
        if self.library_manager.current_library:
            try:
                band_group = self.library_manager.current_library[band]
                if pattern_id in band_group:
                    pattern_group = band_group[pattern_id]

                    # Update metadata label
                    if "metadata" in pattern_group:
                        metadata_group = pattern_group["metadata"]
                        metadata_group.attrs["label"] = actual_label
                        metadata_group.attrs["labeled"] = datetime.now().isoformat()

                    # Record labeling action
                    self.label_history.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "pattern_id": pattern_id,
                            "band": band,
                            "label": actual_label,
                            "action": "labeled",
                        }
                    )

                    return True
            except Exception as e:
                print(f"Error labeling pattern: {e}")

        return False

    def _get_custom_labels(self) -> List[str]:
        """Get custom labels from current library"""
        custom_labels = []

        if self.library_manager.current_library:
            for band_name in self.library_manager.current_library.keys():
                if not isinstance(
                    self.library_manager.current_library[band_name], h5py.Group
                ):
                    continue

                band_group = self.library_manager.current_library[band_name]
                for pattern_name in band_group.keys():
                    if not pattern_name.startswith("pattern_"):
                        continue

                    pattern_group = band_group[pattern_name]
                    metadata = dict(pattern_group.get("metadata", {}).attrs)
                    label = metadata.get("label", "unlabeled")

                    if label not in PREDEFINED_LABELS and label not in custom_labels:
                        custom_labels.append(label)

        return custom_labels

    def get_labeling_stats(self) -> Dict[str, Any]:
        """Get labeling statistics"""
        stats = {
            "total_labeled": len(self.label_history),
            "predefined_labels": {},
            "custom_labels": {},
            "recent_activity": [],
        }

        # Count labels
        for entry in self.label_history:
            label = entry["label"]
            if label in PREDEFINED_LABELS:
                stats["predefined_labels"][label] = (
                    stats["predefined_labels"].get(label, 0) + 1
                )
            else:
                stats["custom_labels"][label] = stats["custom_labels"].get(label, 0) + 1

        # Recent activity (last 10)
        stats["recent_activity"] = self.label_history[-10:]

        return stats


# Import ISM_BANDS from config for HDF5 metadata
from config import ISM_BANDS
