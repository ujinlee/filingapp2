import requests
import json
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def test_sec_api():
    """Test SEC API connectivity"""
    print("\n=== Testing SEC API ===")
    try:
        # Test with Apple's CIK
        response = requests.get(
            "https://data.sec.gov/submissions/CIK0000320193.json",
            headers={"User-Agent": f"Financial Filing Podcast Summarizer {os.getenv('SEC_USER_AGENT_EMAIL')}"}
        )
        if response.status_code == 200:
            print("✅ SEC API connection successful")
            return True
        else:
            print(f"❌ SEC API connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ SEC API error: {str(e)}")
        return False

def test_backend_api():
    """Test FastAPI backend"""
    print("\n=== Testing Backend API ===")
    try:
        # Test with Apple's 10-K
        request_data = {
            "company_ticker": "AAPL",
            "filing_type": "10-K",
            "year": 2023,
            "language": "en-US"
        }
        
        response = requests.post(
            "http://localhost:8000/api/summarize",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backend API test successful")
            print(f"Summary length: {len(result['summary'])} characters")
            print(f"Transcript length: {len(result['transcript'])} characters")
            print(f"Audio URL: {result['audio_url']}")
            return True
        else:
            print(f"❌ Backend API test failed: {response.status_code}")
            print(f"Error: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Backend API error: {str(e)}")
        return False

def test_streamlit():
    """Test Streamlit frontend"""
    print("\n=== Testing Streamlit Frontend ===")
    try:
        response = requests.get("http://localhost:8501")
        if response.status_code == 200:
            print("✅ Streamlit frontend is running")
            return True
        else:
            print(f"❌ Streamlit frontend test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Streamlit frontend error: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Starting Financial Filing Podcast Summarizer Tests...")
    
    # Check environment variables
    print("\n=== Checking Environment Variables ===")
    required_vars = ["GOOGLE_API_KEY", "SEC_USER_AGENT_EMAIL"]
    env_ok = True
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ Missing environment variable: {var}")
            env_ok = False
        else:
            print(f"✅ Found environment variable: {var}")
    
    if not env_ok:
        print("\n❌ Please set all required environment variables before running tests")
        return
    
    # Run tests
    tests = [
        ("SEC API", test_sec_api),
        ("Backend API", test_backend_api),
        ("Streamlit Frontend", test_streamlit)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    run_all_tests() 