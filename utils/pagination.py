# utils/pagination.py - Pydantic v2 Compatible
"""
Pagination and Filtering Utilities for Production API - Pydantic v2
====================================================

Provides standardized pagination, sorting, and filtering across all endpoints.
"""

from typing import Optional, List, Dict, Any, Generic, TypeVar
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Query
from sqlalchemy import desc, asc
from datetime import datetime, date
import math

T = TypeVar('T')

class PaginationParams(BaseModel):
    """Standard pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    limit: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")
    
    @field_validator('page')
    @classmethod
    def validate_page(cls, v):
        if v < 1:
            raise ValueError('Page must be >= 1')
        return v
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Limit must be between 1 and 100')
        return v

class DateRangeParams(BaseModel):
    """Date range filtering parameters"""
    start_date: Optional[date] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[date] = Field(default=None, description="End date (YYYY-MM-DD)")
    
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        if v and 'start_date' in info.data and info.data['start_date']:
            if v < info.data['start_date']:
                raise ValueError('End date must be after start date')
        return v

class HoldingsFilterParams(BaseModel):
    """Holdings-specific filtering parameters"""
    min_value: Optional[float] = Field(default=None, ge=0, description="Minimum holding value")
    max_value: Optional[float] = Field(default=None, ge=0, description="Maximum holding value")
    sector: Optional[str] = Field(default=None, description="Filter by sector")
    asset_type: Optional[str] = Field(default=None, description="Filter by asset type")
    search: Optional[str] = Field(default=None, min_length=1, max_length=50, description="Search ticker or name")

class PaginatedResponse(BaseModel, Generic[T]):
    """Standardized paginated response format"""
    items: List[T]
    total_items: int
    total_pages: int
    current_page: int
    items_per_page: int
    has_next: bool
    has_previous: bool
    next_page: Optional[int] = None
    previous_page: Optional[int] = None

def create_paginated_response(
    items: List[T],
    total_items: int,
    page: int,
    limit: int
) -> PaginatedResponse[T]:
    """
    Create standardized paginated response
    
    Args:
        items: List of items for current page
        total_items: Total number of items across all pages
        page: Current page number
        limit: Items per page
        
    Returns:
        PaginatedResponse with pagination metadata
    """
    total_pages = math.ceil(total_items / limit) if total_items > 0 else 1
    
    return PaginatedResponse(
        items=items,
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        items_per_page=limit,
        has_next=page < total_pages,
        has_previous=page > 1,
        next_page=page + 1 if page < total_pages else None,
        previous_page=page - 1 if page > 1 else None
    )

def apply_pagination(query: Query, page: int, limit: int) -> tuple[Query, int]:
    """
    Apply pagination to SQLAlchemy query
    
    Args:
        query: SQLAlchemy query object
        page: Page number (1-based)
        limit: Items per page
        
    Returns:
        Tuple of (paginated_query, total_count)
    """
    # Get total count before applying pagination
    total_count = query.count()
    
    # Apply pagination
    offset = (page - 1) * limit
    paginated_query = query.offset(offset).limit(limit)
    
    return paginated_query, total_count

def apply_sorting(query: Query, sort_by: Optional[str], sort_order: str, model_class) -> Query:
    """
    Apply sorting to SQLAlchemy query
    
    Args:
        query: SQLAlchemy query object
        sort_by: Field name to sort by
        sort_order: 'asc' or 'desc'
        model_class: SQLAlchemy model class
        
    Returns:
        Query with sorting applied
    """
    if not sort_by:
        return query
    
    # Validate sort field exists on model
    if not hasattr(model_class, sort_by):
        raise ValueError(f"Invalid sort field: {sort_by}")
    
    sort_column = getattr(model_class, sort_by)
    
    if sort_order == 'asc':
        return query.order_by(asc(sort_column))
    else:
        return query.order_by(desc(sort_column))

def apply_date_filter(query: Query, date_field, start_date: Optional[date], end_date: Optional[date]) -> Query:
    """
    Apply date range filtering to query
    
    Args:
        query: SQLAlchemy query object
        date_field: Model field to filter on
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        
    Returns:
        Query with date filtering applied
    """
    if start_date:
        query = query.filter(date_field >= start_date)
    
    if end_date:
        # Add one day to make end_date inclusive
        end_datetime = datetime.combine(end_date, datetime.max.time())
        query = query.filter(date_field <= end_datetime)
    
    return query

def apply_search_filter(query: Query, search_fields: List, search_term: Optional[str]) -> Query:
    """
    Apply text search filtering to query
    
    Args:
        query: SQLAlchemy query object
        search_fields: List of model fields to search
        search_term: Search term
        
    Returns:
        Query with search filtering applied
    """
    if not search_term:
        return query
    
    search_term = f"%{search_term.lower()}%"
    
    # Create OR conditions for all search fields
    conditions = []
    for field in search_fields:
        conditions.append(field.ilike(search_term))
    
    if conditions:
        from sqlalchemy import or_
        query = query.filter(or_(*conditions))
    
    return query

def validate_sort_fields(sort_by: Optional[str], allowed_fields: List[str]) -> str:
    """
    Validate and sanitize sort field
    
    Args:
        sort_by: Requested sort field
        allowed_fields: List of allowed sort fields
        
    Returns:
        Validated sort field
        
    Raises:
        ValueError: If sort field is not allowed
    """
    if not sort_by:
        return allowed_fields[0] if allowed_fields else "id"
    
    if sort_by not in allowed_fields:
        raise ValueError(f"Invalid sort field. Allowed fields: {', '.join(allowed_fields)}")
    
    return sort_by

class FilterValidator:
    """Utility class for validating filter parameters"""
    
    @staticmethod
    def validate_numeric_range(min_val: Optional[float], max_val: Optional[float], field_name: str):
        """Validate numeric range parameters"""
        if min_val is not None and min_val < 0:
            raise ValueError(f"{field_name} minimum value must be >= 0")
        
        if max_val is not None and max_val < 0:
            raise ValueError(f"{field_name} maximum value must be >= 0")
        
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"{field_name} minimum value cannot be greater than maximum value")
    
    @staticmethod
    def sanitize_search_term(search_term: Optional[str]) -> Optional[str]:
        """Sanitize search term for SQL injection protection"""
        if not search_term:
            return None
        
        # Remove potentially dangerous characters
        sanitized = search_term.strip()
        # Remove SQL injection patterns
        dangerous_patterns = ['--', ';', '/*', '*/', 'xp_', 'sp_', 'DROP', 'DELETE', 'INSERT', 'UPDATE']
        
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        return sanitized if sanitized else None