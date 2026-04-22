import imaplib
import email
import time
import json
import requests
from email.header import decode_header
import os

# ====================================================
# YOUR GMAIL ACCOUNT SETTINGS
# ====================================================

EMAIL_ADDRESS = "glitchm917@gmail.com"
EMAIL_PASSWORD = "bqzf lzzu jgsm qxjz"  # Your App Password
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# Your ngrok webhook URL
WEBHOOK_URL = "https://bleep-latitude-morbidly.ngrok-free.dev/webhook"

# Check interval (seconds)
CHECK_INTERVAL = 5

# ====================================================a
# DO NOT MODIFY BELOW THIS LINE
# ====================================================

processed_ids = set()

def get_email_body(msg):
    """Extract body from email"""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                try:
                    body = part.get_payload(decode=True).decode()
                    return body
                except:
                    return ""
    else:
        try:
            body = msg.get_payload(decode=True).decode()
            return body
        except:
            return ""
    return ""

def send_to_webhook(data):
    """Send email content to your Python server"""
    try:
        response = requests.post(WEBHOOK_URL, json=data, timeout=5)
        print(f"   ✅ Forwarded to webhook: {response.status_code}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to send to webhook: {e}")
        return False

def check_emails():
    """Check for new emails from TradingView"""
    try:
        # Connect to email server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("INBOX")
        
        # Search for unseen emails
        status, messages = mail.search(None, 'UNSEEN')
        
        if status == "OK":
            email_ids = messages[0].split()
            
            for email_id in email_ids:
                if email_id in processed_ids:
                    continue
                    
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                
                if status == "OK":
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # Check if from TradingView
                            from_addr = msg.get("From", "")
                            subject = msg.get("Subject", "")
                            
                            if "tradingview" in from_addr.lower() or "alert" in subject.lower():
                                body = get_email_body(msg)
                                
                                print(f"\n📧 New TradingView alert!")
                                print(f"   📧 Account: {EMAIL_ADDRESS}")
                                print(f"   📋 Subject: {subject[:100]}")
                                print(f"   📄 Body: {body[:200]}...")
                                
                                # Forward to webhook
                                webhook_data = {
                                    "token": "MySuperSecretKey123",
                                    "raw_email": body,
                                    "subject": subject,
                                    "source": EMAIL_ADDRESS
                                }
                                send_to_webhook(webhook_data)
                                
                                processed_ids.add(email_id)
        
        mail.close()
        mail.logout()
        
    except Exception as e:
        print(f"❌ Email check error: {e}")

def main():
    print("="*60)
    print("📧 EMAIL TO WEBHOOK FORWARDER")
    print("="*60)
    print(f"📡 Monitoring: {EMAIL_ADDRESS}")
    print(f"🔗 Forwarding to: {WEBHOOK_URL}")
    print(f"⏱️  Check interval: {CHECK_INTERVAL} seconds")
    print("="*60)
    print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            check_emails()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\n👋 Stopping email listener...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()