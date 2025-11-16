#!/usr/bin/env python
"""
Test script to verify MCMulT installation and basic functionality
Run this script to ensure all components are working correctly
"""

import sys
import os

def check_imports():
    """Check if all required packages are installed"""
    print("Checking imports...")
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'h5py': 'h5py'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\nAll required packages installed!")
    return True


def check_modules():
    """Check if MCMulT modules can be imported"""
    print("\nChecking MCMulT modules...")
    modules = {
        'mcmult_model': 'Model architecture',
        'data_loader_hpc': 'Data loader',
        'train_hpc': 'Training script',
        'utils_hpc': 'Utilities'
    }
    
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"  ✓ {description} ({module}.py)")
        except Exception as e:
            print(f"  ✗ {description} ({module}.py) - {e}")
            return False
    
    print("\nAll modules loaded successfully!")
    return True


def test_model():
    """Test model creation and forward pass"""
    print("\nTesting model creation...")
    
    try:
        import torch
        from mcmult_model import MCMulT
        
        # Create model
        model = MCMulT(
            audio_dim=74,
            visual_dim=35,
            text_dim=300,
            d_model=64,
            nhead=4,
            num_layers=2
        )
        
        print(f"  ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        
        audio = torch.randn(batch_size, seq_len, 74)
        visual = torch.randn(batch_size, seq_len, 35)
        text = torch.randn(batch_size, seq_len, 300)
        
        output = model(audio, visual, text)
        
        assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
        print(f"  ✓ Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_data_loader():
    """Test data loader"""
    print("\nTesting data loader...")
    
    try:
        from data_loader_hpc import CMUMOSIDataset
        
        # Create dataset
        dataset = CMUMOSIDataset('./data/mosi', split='train', max_seq_len=50)
        
        print(f"  ✓ Dataset created with {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        
        assert 'audio' in sample, "Missing 'audio' key"
        assert 'visual' in sample, "Missing 'visual' key"
        assert 'text' in sample, "Missing 'text' key"
        assert 'label' in sample, "Missing 'label' key"
        
        print(f"  ✓ Sample loaded: audio={sample['audio'].shape}, "
              f"visual={sample['visual'].shape}, text={sample['text'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data loader test failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        
        config_file = 'config.yaml'
        if not os.path.exists(config_file):
            print(f"  ✗ {config_file} not found")
            return False
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = [
            'data_path', 'audio_dim', 'visual_dim', 'text_dim',
            'd_model', 'nhead', 'num_layers', 'batch_size',
            'num_epochs', 'learning_rate', 'seed'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"  ✗ Missing config keys: {', '.join(missing_keys)}")
            return False
        
        print(f"  ✓ Configuration loaded successfully")
        print(f"    Model: d_model={config['d_model']}, nhead={config['nhead']}, layers={config['num_layers']}")
        print(f"    Training: batch_size={config['batch_size']}, epochs={config['num_epochs']}, lr={config['learning_rate']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def test_utils():
    """Test utilities"""
    print("\nTesting utilities...")
    
    try:
        from utils_hpc import set_seed, calculate_metrics
        import numpy as np
        
        # Test seeding
        set_seed(42)
        print(f"  ✓ Deterministic seeding configured")
        
        # Test metrics calculation
        predictions = np.random.randn(100)
        labels = np.random.randn(100)
        metrics = calculate_metrics(predictions, labels)
        
        assert 'ccc' in metrics, "Missing CCC metric"
        assert 'mae' in metrics, "Missing MAE metric"
        assert 'corr' in metrics, "Missing correlation metric"
        
        print(f"  ✓ Metrics calculation working")
        print(f"    Available metrics: {', '.join(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Utils test failed: {e}")
        return False


def main():
    """Main test function"""
    print("="*70)
    print("MCMulT Installation and Functionality Test")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Package imports", check_imports()))
    results.append(("Module imports", check_modules()))
    results.append(("Model creation", test_model()))
    results.append(("Data loader", test_data_loader()))
    results.append(("Configuration", test_config()))
    results.append(("Utilities", test_utils()))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! MCMulT is ready to use.")
        print("\nNext steps:")
        print("  1. Prepare CMU-MOSI dataset in data/mosi/")
        print("  2. Adjust config.yaml for your setup")
        print("  3. Run training: python train_hpc.py --config config.yaml")
        print("  4. Or use quick start: ./quick_start.sh")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
