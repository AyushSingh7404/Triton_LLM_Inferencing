#!/usr/bin/env python3
import requests
import json
import sys
import time

def check_server_health():
    """Check Triton server health"""
    try:
        # Check server ready
        response = requests.get("http://localhost:8000/v2/health/ready")
        if response.status_code != 200:
            print(f"Server not ready: {response.status_code}")
            return False
        
        # Check server live
        response = requests.get("http://localhost:8000/v2/health/live")
        if response.status_code != 200:
            print(f"Server not live: {response.status_code}")
            return False
        
        # Check model ready
        response = requests.get("http://localhost:8000/v2/models/mistral-7b/ready")
        if response.status_code != 200:
            print(f"Model not ready: {response.status_code}")
            return False
        
        print("✅ Server and model are healthy")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def get_server_metadata():
    """Get server metadata"""
    try:
        response = requests.get("http://localhost:8000/v2")
        if response.status_code == 200:
            data = response.json()
            print(f"Server name: {data['name']}")
            print(f"Server version: {data['version']}")
        
        response = requests.get("http://localhost:8000/v2/models/mistral-7b")
        if response.status_code == 200:
            data = response.json()
            print(f"Model name: {data['name']}")
            print(f"Model versions: {data['versions']}")
            print(f"Model platform: {data['platform']}")
    except Exception as e:
        print(f"Error getting metadata: {e}")

if __name__ == "__main__":
    if check_server_health():
        get_server_metadata()
        sys.exit(0)
    else:
        sys.exit(1)