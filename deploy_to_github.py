#!/usr/bin/env python3
"""
Deployment Script for Customer Churn Prediction System
This script helps set up GitHub repository and deploy the application
"""
import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_application_status():
    """Check if the FastAPI application is running"""
    print("\n🔍 Checking application status...")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI application is running on http://localhost:8000")
            print("📚 API documentation available at: http://localhost:8000/docs")
            return True
        else:
            print(f"⚠️ Application responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Application is not accessible: {e}")
        return False

def setup_github_repo():
    """Set up GitHub repository"""
    print("\n🚀 Setting up GitHub repository...")
    
    # Get GitHub username
    username = input("Enter your GitHub username: ").strip()
    if not username:
        print("❌ Username is required")
        return False
    
    # Set remote origin
    remote_url = f"https://github.com/{username}/churn_detection.git"
    print(f"Setting remote origin to: {remote_url}")
    
    # Remove existing remote if any
    run_command("git remote remove origin", "Removing existing remote")
    
    # Add new remote
    result = run_command(f'git remote add origin {remote_url}', "Adding GitHub remote")
    if not result:
        return False
    
    # Push to GitHub
    print("\n📤 Pushing code to GitHub...")
    print("Note: You may need to create the repository on GitHub first")
    print("Go to: https://github.com/new")
    print("Repository name: churn_detection")
    print("Make it public or private as per your preference")
    
    input("\nPress Enter after creating the repository on GitHub...")
    
    # Push the code
    result = run_command("git push -u origin master", "Pushing code to GitHub")
    if result:
        print(f"✅ Code successfully pushed to: {remote_url}")
        return True
    else:
        print("❌ Failed to push code to GitHub")
        return False

def start_application():
    """Start the FastAPI application if not running"""
    print("\n🚀 Starting FastAPI application...")
    
    if check_application_status():
        print("✅ Application is already running")
        return True
    
    # Check if we have the required dependencies
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("❌ FastAPI or uvicorn not installed")
        print("Installing required dependencies...")
        run_command("pip install -r requirements.txt", "Installing dependencies")
    
    # Start the application
    print("Starting FastAPI application...")
    try:
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for the app to start
        time.sleep(3)
        
        if check_application_status():
            print("✅ Application started successfully")
            return True
        else:
            print("❌ Failed to start application")
            return False
            
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Customer Churn Prediction System - Deployment Script")
    print("=" * 60)
    
    # Check application status
    app_running = check_application_status()
    
    if not app_running:
        print("\n⚠️ Application is not running. Starting it now...")
        if not start_application():
            print("❌ Failed to start application. Please check the logs.")
            return
    
    # Setup GitHub repository
    print("\n📚 Setting up GitHub repository...")
    if setup_github_repo():
        print("\n🎉 Deployment completed successfully!")
        print("\n📋 Next steps:")
        print("1. Your application is running at: http://localhost:8000")
        print("2. API documentation: http://localhost:8000/docs")
        print("3. Code is now on GitHub: https://github.com/[username]/churn_detection")
        print("4. You can deploy to cloud platforms like:")
        print("   - Heroku")
        print("   - Railway")
        print("   - Render")
        print("   - AWS/GCP/Azure")
    else:
        print("\n❌ GitHub setup failed. Please try again manually.")

if __name__ == "__main__":
    main()
