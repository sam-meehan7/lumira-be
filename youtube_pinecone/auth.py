from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from jwt import PyJWTError

security = HTTPBearer()

# Replace with your Supabase JWT secret
JWT_SECRET = "eKF7wZ9BNU8IkmgdZZKUYBaB+U/LQHZLyK94stbv7zsYkOosMeefLjSPpIecjZveOuEqjuO2aJWjcUjvd0ty0A=="

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        token = credentials.credentials
        # Add logging to see the token being processed
        print(f"Processing token: {token[:20]}...")  # Only print start of token for security

        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            # Add these options for Supabase JWT validation
            options={"verify_exp": True, "verify_aud": False}
        )

        # Add logging to see the decoded payload
        print(f"Decoded payload: {payload}")

        # Check for user ID in both 'sub' and 'user_id' fields (Supabase uses 'sub')
        user_id = payload.get("sub") or payload.get("user_id")
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="User ID not found in token"
            )
        return user_id
    except PyJWTError as e:
        # Add more detailed error message
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication token: {str(e)}"
        )