# backend/database/database.py
''' SQLAlchemy is the Python SQL toolkit that allows developers to access and manage SQL databases using Pythonic domain language.
 Its primary role is to help Python applications interact with relational databases by making it easier to create, update, and query tables.
 -declarative_base: combines a metadata container and a mapper that maps our class to a database table
 '''

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.config import settings

#The create_engine()  plays a main role. It acts as a gateway to set up a connection between our Python application and a PostgreSQL database.
engine = create_engine(settings.DATABASE_URL)

# sessionmaker: to create a custom Session class tailored to your application's needs
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base() # Base class for our ORM models
# The Base class serves as a foundation for defining our database models.
