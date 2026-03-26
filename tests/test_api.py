import requests
import os

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(BASE_URL)
    assert response.status_code == 200
    print("Health check passed.")

def test_history():
    response = requests.get(f"{BASE_URL}/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print(f"History check passed. Found {len(data)} items.")

if __name__ == "__main__":
    try:
        test_health()
        test_history()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
