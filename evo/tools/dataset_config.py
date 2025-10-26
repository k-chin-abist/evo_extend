#!/usr/bin/env python3
"""
Dataset Configuration Manager for EVO
"""

import json
from pathlib import Path


class DatasetConfig:
    """Manage dataset root directory configuration"""
    
    # Config file location
    CONFIG_FILE = Path.home() / ".evo" / "dataset_config.json"
    
    # Supported datasets
    SUPPORTED_DATASETS = ["kitti", "tum", "euroc"]
    
    def __init__(self):
        """Initialize config manager"""
        self._ensure_config_exists()
        self._load_config()
    
    def _ensure_config_exists(self):
        """Ensure config file exists"""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.CONFIG_FILE.exists():
            # Create empty config
            default_config = {name: None for name in self.SUPPORTED_DATASETS}
            self._save_config(default_config)
    
    def _load_config(self):
        """Load config"""
        with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def _save_config(self, config=None):
        """Save config"""
        if config is None:
            config = self.config
        
        with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def set_root(self, dataset_name, root_path):
        """
        Set dataset root directory
        
        Args:
            dataset_name: Dataset name (kitti, tum, euroc)
            root_path: Root directory path
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}\n"
                f"Supported datasets: {', '.join(self.SUPPORTED_DATASETS)}"
            )
        
        # Validate path exists
        path = Path(root_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        # Save config
        self.config[dataset_name] = str(path)
        self._save_config()
        
        print(f"OK Set {dataset_name.upper()} root: {path}")
    
    def get_root(self, dataset_name):
        """
        Get dataset root directory
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Path: Root directory path
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self.config:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        root = self.config[dataset_name]
        
        if root is None:
            raise ValueError(
                f"{dataset_name.upper()} root not configured\n"
                f"Use: config.set_root('{dataset_name}', '/your/path')"
            )
        
        return Path(root)
    
    def get_sequence_path(self, dataset_name, sequence_id):
        """
        Get sequence path
        
        Args:
            dataset_name: Dataset name
            sequence_id: Sequence ID (e.g., "00", "01" or "freiburg1_xyz")
            
        Returns:
            Path: Full sequence path
        """
        root = self.get_root(dataset_name)
        dataset_name = dataset_name.lower()
        
        if dataset_name == "kitti":
            # KITTI: sequences/00, sequences/01, ...
            seq_num = int(sequence_id) if isinstance(sequence_id, str) else sequence_id
            return root / "sequences" / f"{seq_num:02d}"
        
        elif dataset_name in ["tum", "euroc"]:
            # TUM/EuRoC: use sequence name directly
            return root / sequence_id
        
        else:
            return root / sequence_id
    
    def show_config(self):
        """Show current configuration"""
        print("\n" + "=" * 60)
        print("Dataset Configuration")
        print("=" * 60)
        print(f"Config file: {self.CONFIG_FILE}\n")
        
        for dataset_name in self.SUPPORTED_DATASETS:
            root = self.config.get(dataset_name)
            status = "OK" if root else "NO"
            value = root if root else "<not configured>"
            print(f"  {status} {dataset_name.upper():8s}: {value}")
        
        print("=" * 60 + "\n")
    
    def clear_root(self, dataset_name):
        """Clear dataset root configuration"""
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.config[dataset_name] = None
        self._save_config()
        
        print(f"OK Cleared {dataset_name.upper()} root configuration")


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EVO Dataset Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set KITTI root directory
  python dataset_settings.py set kitti /path/to/KITTI
  
  # Set TUM root directory  
  python dataset_settings.py set tum /path/to/TUM
  
  # Show current configuration
  python dataset_settings.py show
  
  # Clear configuration
  python dataset_settings.py clear kitti
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='command')
    
    # set command
    set_parser = subparsers.add_parser('set', help='Set dataset root directory')
    set_parser.add_argument('dataset', choices=['kitti', 'tum', 'euroc'], help='Dataset name')
    set_parser.add_argument('path', help='Root directory path')
    
    # show command
    subparsers.add_parser('show', help='Show current configuration')
    
    # clear command
    clear_parser = subparsers.add_parser('clear', help='Clear dataset configuration')
    clear_parser.add_argument('dataset', choices=['kitti', 'tum', 'euroc'], help='Dataset name')
    
    args = parser.parse_args()
    
    config = DatasetConfig()
    
    if args.command == 'set':
        config.set_root(args.dataset, args.path)
    elif args.command == 'show':
        config.show_config()
    elif args.command == 'clear':
        config.clear_root(args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()