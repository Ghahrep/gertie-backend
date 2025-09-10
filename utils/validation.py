# utils/validation.py - Request Validation and Sanitization - Pydantic v2
"""
Request Validation and Business Rule Validation - Pydantic v2
==============================================

Provides comprehensive input validation and business rule enforcement.
"""

import re
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal, InvalidOperation
from datetime import datetime, date
from pydantic import BaseModel, field_validator
from fastapi import HTTPException, status

class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(message)

class BusinessRuleError(Exception):
    """Custom business rule validation error"""
    def __init__(self, message: str, rule: str = None):
        self.message = message
        self.rule = rule
        super().__init__(message)

class InputSanitizer:
    """Utility class for input sanitization"""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 255, allow_special_chars: bool = True) -> str:
        """
        Sanitize string input for SQL injection and XSS protection
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            allow_special_chars: Whether to allow special characters
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")
        
        # Trim whitespace
        sanitized = value.strip()
        
        # Check length
        if len(sanitized) > max_length:
            raise ValidationError(f"Input too long. Maximum {max_length} characters allowed")
        
        if not sanitized:
            raise ValidationError("Input cannot be empty")
        
        # Remove potential XSS patterns
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
        ]
        
        for pattern in xss_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Remove SQL injection patterns
        sql_patterns = [
            r';\s*DROP\s+TABLE',
            r';\s*DELETE\s+FROM',
            r';\s*INSERT\s+INTO',
            r';\s*UPDATE\s+',
            r'UNION\s+SELECT',
            r'--\s*',
            r'/\*.*?\*/',
        ]
        
        for pattern in sql_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Restrict special characters if not allowed
        if not allow_special_chars:
            if not re.match(r'^[a-zA-Z0-9\s\-_.]+$', sanitized):
                raise ValidationError("Input contains invalid characters")
        
        return sanitized
    
    @staticmethod
    def sanitize_email(email: str) -> str:
        """Sanitize and validate email address"""
        if not isinstance(email, str):
            raise ValidationError("Email must be a string")
        
        email = email.strip().lower()
        
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            raise ValidationError("Invalid email format")
        
        if len(email) > 254:  # RFC 5321 limit
            raise ValidationError("Email address too long")
        
        return email
    
    @staticmethod
    def sanitize_ticker(ticker: str) -> str:
        """Sanitize stock ticker symbol"""
        if not isinstance(ticker, str):
            raise ValidationError("Ticker must be a string")
        
        ticker = ticker.strip().upper()
        
        # Ticker validation - alphanumeric only, 1-6 characters
        if not re.match(r'^[A-Z0-9]{1,6}$', ticker):
            raise ValidationError("Invalid ticker format. Must be 1-6 alphanumeric characters")
        
        return ticker

class FinancialValidator:
    """Validator for financial data and business rules"""
    
    @staticmethod
    def validate_amount(amount: Union[float, int, str], min_value: float = 0, max_value: float = None) -> float:
        """
        Validate monetary amount
        
        Args:
            amount: Amount to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated amount as float
            
        Raises:
            ValidationError: If amount is invalid
        """
        try:
            if isinstance(amount, str):
                # Remove currency symbols and commas
                amount = re.sub(r'[$,]', '', amount)
                amount = float(amount)
            else:
                amount = float(amount)
        except (ValueError, TypeError):
            raise ValidationError("Invalid amount format")
        
        if amount < min_value:
            raise ValidationError(f"Amount must be at least ${min_value:,.2f}")
        
        if max_value is not None and amount > max_value:
            raise ValidationError(f"Amount cannot exceed ${max_value:,.2f}")
        
        # Check for reasonable decimal places (2 for currency)
        if round(amount, 2) != amount:
            amount = round(amount, 2)
        
        return amount
    
    @staticmethod
    def validate_percentage(percentage: Union[float, int, str], allow_negative: bool = False) -> float:
        """
        Validate percentage value
        
        Args:
            percentage: Percentage to validate
            allow_negative: Whether negative percentages are allowed
            
        Returns:
            Validated percentage as decimal (0.15 for 15%)
            
        Raises:
            ValidationError: If percentage is invalid
        """
        try:
            if isinstance(percentage, str):
                # Remove % symbol
                percentage = percentage.replace('%', '')
                percentage = float(percentage)
            else:
                percentage = float(percentage)
        except (ValueError, TypeError):
            raise ValidationError("Invalid percentage format")
        
        # Convert from percentage to decimal if > 1
        if percentage > 1:
            percentage = percentage / 100
        
        if not allow_negative and percentage < 0:
            raise ValidationError("Percentage cannot be negative")
        
        if percentage > 1:
            raise ValidationError("Percentage cannot exceed 100%")
        
        return round(percentage, 4)  # 4 decimal places for precision
    
    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate portfolio allocation weights
        
        Args:
            weights: Dictionary of ticker -> weight
            
        Returns:
            Validated and normalized weights
            
        Raises:
            BusinessRuleError: If weights don't meet business rules
        """
        if not weights:
            raise BusinessRuleError("Portfolio weights cannot be empty")
        
        # Validate individual weights
        validated_weights = {}
        total_weight = 0
        
        for ticker, weight in weights.items():
            # Sanitize ticker
            clean_ticker = InputSanitizer.sanitize_ticker(ticker)
            
            # Validate weight
            clean_weight = FinancialValidator.validate_percentage(weight)
            
            if clean_weight <= 0:
                raise BusinessRuleError(f"Weight for {clean_ticker} must be positive")
            
            validated_weights[clean_ticker] = clean_weight
            total_weight += clean_weight
        
        # Check total weights
        if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
            raise BusinessRuleError(f"Portfolio weights must sum to 100% (current: {total_weight*100:.1f}%)")
        
        # Normalize weights to exactly 1.0
        if total_weight != 1.0:
            for ticker in validated_weights:
                validated_weights[ticker] = validated_weights[ticker] / total_weight
        
        return validated_weights
    
    @staticmethod
    def validate_date_range(start_date: Optional[date], end_date: Optional[date]) -> tuple[Optional[date], Optional[date]]:
        """
        Validate date range for analysis
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Validated date range
            
        Raises:
            ValidationError: If date range is invalid
        """
        today = date.today()
        
        if start_date and start_date > today:
            raise ValidationError("Start date cannot be in the future")
        
        if end_date and end_date > today:
            raise ValidationError("End date cannot be in the future")
        
        if start_date and end_date and start_date > end_date:
            raise ValidationError("Start date cannot be after end date")
        
        # Business rule: Maximum 10 years of historical data
        if start_date:
            from datetime import timedelta
            max_start_date = today - timedelta(days=10*365)
            if start_date < max_start_date:
                raise BusinessRuleError("Analysis limited to 10 years of historical data")
        
        return start_date, end_date

class PortfolioBusinessRules:
    """Business rule validators for portfolio operations"""
    
    @staticmethod
    def validate_trade_order(
        ticker: str,
        action: str,
        amount: float,
        portfolio_value: float
    ):
        """
        Validate trade order business rules
        
        Args:
            ticker: Stock ticker
            action: 'BUY' or 'SELL'
            amount: Trade amount in USD
            portfolio_value: Total portfolio value
            
        Raises:
            BusinessRuleError: If trade violates business rules
        """
        # Sanitize inputs
        clean_ticker = InputSanitizer.sanitize_ticker(ticker)
        clean_amount = FinancialValidator.validate_amount(amount, min_value=1)
        
        if action not in ['BUY', 'SELL']:
            raise ValidationError("Action must be 'BUY' or 'SELL'")
        
        # Business rule: Minimum trade amount
        if clean_amount < 100:
            raise BusinessRuleError("Minimum trade amount is $100")
        
        # Business rule: Maximum single trade size (20% of portfolio)
        max_trade_amount = portfolio_value * 0.2
        if clean_amount > max_trade_amount:
            raise BusinessRuleError(f"Single trade cannot exceed 20% of portfolio value (${max_trade_amount:,.2f})")
        
        # Business rule: Maximum position size (40% of portfolio)
        max_position_size = portfolio_value * 0.4
        if action == 'BUY' and clean_amount > max_position_size:
            raise BusinessRuleError(f"Position size cannot exceed 40% of portfolio value (${max_position_size:,.2f})")
    
    @staticmethod
    def validate_rebalancing_frequency(last_rebalance_date: Optional[datetime]):
        """
        Validate rebalancing frequency business rules
        
        Args:
            last_rebalance_date: Date of last rebalancing
            
        Raises:
            BusinessRuleError: If rebalancing too frequent
        """
        if last_rebalance_date:
            from datetime import timedelta
            min_rebalance_interval = timedelta(days=30)  # Minimum 30 days between rebalancing
            
            if datetime.now() - last_rebalance_date < min_rebalance_interval:
                days_remaining = (last_rebalance_date + min_rebalance_interval - datetime.now()).days
                raise BusinessRuleError(f"Rebalancing too frequent. Wait {days_remaining} more days")

def validate_request_data(data: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic request data validation
    
    Args:
        data: Request data to validate
        validation_rules: Validation rules to apply
        
    Returns:
        Validated and sanitized data
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        validated_data = {}
        
        for field, rules in validation_rules.items():
            if field not in data:
                if rules.get('required', False):
                    raise ValidationError(f"Field '{field}' is required")
                continue
            
            value = data[field]
            field_type = rules.get('type', 'string')
            
            # Type-specific validation
            if field_type == 'string':
                validated_data[field] = InputSanitizer.sanitize_string(
                    value, 
                    max_length=rules.get('max_length', 255),
                    allow_special_chars=rules.get('allow_special_chars', True)
                )
            elif field_type == 'email':
                validated_data[field] = InputSanitizer.sanitize_email(value)
            elif field_type == 'ticker':
                validated_data[field] = InputSanitizer.sanitize_ticker(value)
            elif field_type == 'amount':
                validated_data[field] = FinancialValidator.validate_amount(
                    value,
                    min_value=rules.get('min_value', 0),
                    max_value=rules.get('max_value')
                )
            elif field_type == 'percentage':
                validated_data[field] = FinancialValidator.validate_percentage(
                    value,
                    allow_negative=rules.get('allow_negative', False)
                )
            else:
                validated_data[field] = value
        
        return validated_data
        
    except (ValidationError, BusinessRuleError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Request validation failed: {str(e)}"
        )