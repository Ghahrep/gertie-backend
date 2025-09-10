# auth/password_utils.py - Password Hashing Utilities
"""
Secure Password Hashing Utilities
=================================

Production-ready password hashing using bcrypt.
Replace the simple SHA256 hashing in the authentication middleware.
"""

import bcrypt

def hash_password(plain_password: str) -> str:
    """
    Hash a plain password using bcrypt
    
    Args:
        plain_password: Plain text password
        
    Returns:
        str: Hashed password suitable for database storage
    """
    # Convert password to bytes
    password_bytes = plain_password.encode('utf-8')
    
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Return as string for database storage
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database
        
    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        # Convert to bytes
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        
        # Verify password
        return bcrypt.checkpw(password_bytes, hashed_bytes)
        
    except Exception as e:
        # Log error in production
        print(f"Password verification error: {e}")
        return False

def is_password_strong(password: str) -> tuple[bool, list[str]]:
    """
    Check if a password meets security requirements
    
    Args:
        password: Password to check
        
    Returns:
        tuple: (is_strong, list_of_issues)
    """
    issues = []
    
    # Length check
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    # Character type checks
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*(),.?\":{}|<>" for c in password)
    
    if not has_upper:
        issues.append("Password must contain at least one uppercase letter")
    if not has_lower:
        issues.append("Password must contain at least one lowercase letter")
    if not has_digit:
        issues.append("Password must contain at least one number")
    if not has_special:
        issues.append("Password must contain at least one special character")
    
    # Common password check (basic)
    common_passwords = [
        "password", "123456", "password123", "admin", "qwerty",
        "letmein", "welcome", "monkey", "dragon", "master"
    ]
    
    if password.lower() in common_passwords:
        issues.append("Password is too common")
    
    return len(issues) == 0, issues

# Test functions
def test_password_functions():
    """Test the password utility functions"""
    print("Testing password utilities...")
    
    # Test password hashing
    test_password = "TestPassword123!"
    hashed = hash_password(test_password)
    print(f"Original: {test_password}")
    print(f"Hashed: {hashed}")
    
    # Test verification
    is_valid = verify_password(test_password, hashed)
    print(f"Verification: {is_valid}")
    
    # Test wrong password
    is_invalid = verify_password("WrongPassword", hashed)
    print(f"Wrong password: {is_invalid}")
    
    # Test password strength
    strong, issues = is_password_strong(test_password)
    print(f"Password strength: {strong}, Issues: {issues}")
    
    weak_password = "123"
    weak_strong, weak_issues = is_password_strong(weak_password)
    print(f"Weak password strength: {weak_strong}, Issues: {weak_issues}")

if __name__ == "__main__":
    test_password_functions()