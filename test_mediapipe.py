"""
Test script to check MediaPipe installation
Run: python test_mediapipe.py
"""

print("Testing MediaPipe installation...\n")

# Test 1: Import mediapipe
try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
    print(f"  Version: {mp.__version__}")
except ImportError as e:
    print(f"✗ Cannot import mediapipe: {e}")
    print("  Run: pip install mediapipe")
    exit(1)

# Test 2: Check solutions attribute
try:
    solutions = mp.solutions
    print("✓ mp.solutions exists")
except AttributeError:
    print("✗ mp.solutions does NOT exist")
    print("  This is unusual for MediaPipe 0.10.x")

# Test 3: Check hands
try:
    hands = mp.solutions.hands
    print("✓ mp.solutions.hands exists")
except AttributeError as e:
    print(f"✗ mp.solutions.hands does NOT exist: {e}")
    print("\nTrying alternative imports...")
    
    # Try alternatives
    alternatives = [
        "from mediapipe.python.solutions import hands",
        "from mediapipe.solutions import hands",
        "import mediapipe.solutions.hands",
    ]
    
    for alt in alternatives:
        try:
            exec(alt)
            print(f"  ✓ This works: {alt}")
        except Exception as e:
            print(f"  ✗ Failed: {alt}")

# Test 4: List what's in mediapipe
print("\n" + "="*60)
print("Available in mediapipe module:")
print("="*60)
attrs = [attr for attr in dir(mp) if not attr.startswith('_')]
for attr in attrs[:20]:  # Show first 20
    print(f"  - {attr}")
if len(attrs) > 20:
    print(f"  ... and {len(attrs)-20} more")

# Test 5: Try creating Hands object
print("\n" + "="*60)
print("Attempting to create Hands object:")
print("="*60)

try:
    hands_obj = mp.solutions.hands.Hands()
    print("✓ SUCCESS! Hands object created")
    hands_obj.close()
except Exception as e:
    print(f"✗ Failed: {e}")
    print("\nYour MediaPipe installation may need reinstallation:")
    print("  pip uninstall mediapipe -y")
    print("  pip install mediapipe")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
