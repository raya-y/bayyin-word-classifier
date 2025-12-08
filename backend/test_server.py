"""
Quick test script to check if the server is working and models are loaded.
"""
import requests
import json

try:
    # Test health endpoint
    print("Testing server health...")
    response = requests.get("http://localhost:8000/health")
    print(f"Health check: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    response = requests.get("http://localhost:8000/")
    print(f"Root check: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test classification with Arabic text
    print("\nTesting classification...")
    response = requests.post(
        "http://localhost:8000/classify",
        json={"text": "هذا نص عربي للاختبار"}
    )
    print(f"Classification: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to server. Make sure the backend is running on port 8000.")
    print("Start it with: cd backend && python -m app.main")
except Exception as e:
    print(f"ERROR: {str(e)}")

