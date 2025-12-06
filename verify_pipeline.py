#!/usr/bin/env python3
"""
Verification script to test the tourism prediction pipeline
"""
import os
import sys
import subprocess

def run_command(cmd, description, timeout=30):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {description} - PASSED")
            return True
        else:
            print(f"✗ {description} - FAILED (exit code: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False

def main():
    """Main verification function"""
    print("Tourism Prediction Pipeline Verification")
    print("=" * 60)
    
    results = []
    
    # Test 1: Data Registration
    results.append(run_command(
        "python tourism_project/data/data_registration.py",
        "Data Registration"
    ))
    
    # Test 2: Data Preparation
    results.append(run_command(
        "python tourism_project/model_building/data_preparation.py",
        "Data Preparation"
    ))
    
    # Test 3: Model Training
    results.append(run_command(
        "python tourism_project/model_building/model_training.py",
        "Model Training"
    ))
    
    # Test 4: Check generated files
    print(f"\n{'='*60}")
    print("Checking Generated Files")
    print(f"{'='*60}")
    
    files_to_check = [
        "tourism_project/model_building/preprocessing/scaler.pkl",
        "tourism_project/model_building/preprocessing/label_encoders.pkl",
        "tourism_project/model_building/models/tourism_model.pkl",
        "tourism_project/model_building/processed_data/X_train.npy",
        "tourism_project/model_building/processed_data/X_test.npy",
    ]
    
    files_ok = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            files_ok = False
    
    results.append(files_ok)
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Pipeline is working correctly!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Please review the output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
