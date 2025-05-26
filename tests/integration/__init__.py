"""
Integration tests for ReAgent system.

This package contains integration tests that test the interaction
between multiple components with real or minimal mocking.
"""

# Integration test configuration
INTEGRATION_TEST_TIMEOUT = 30  # seconds

# Test categories
REQUIRES_ALL_SERVICES = "requires_all_services"
REQUIRES_REAL_LLM = "requires_real_llm"

# Helper functions for integration tests
async def wait_for_services(timeout: int = 10) -> bool:
    """Wait for all required services to be available."""
    import asyncio
    from db.session import check_database_connection
    import redis
    
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        try:
            # Check database
            db_ok = await check_database_connection()
            
            # Check Redis
            r = redis.Redis(host='localhost', port=6379, db=15)
            redis_ok = r.ping()
            r.close()
            
            if db_ok and redis_ok:
                return True
                
        except Exception:
            pass
            
        await asyncio.sleep(0.5)
    
    return False
