from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Double
from sqlalchemy.orm import relationship

from database import Base


class Topics(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, default="No description")
    is_active = Column(Boolean, default=True)
    intents = relationship("Intents", back_populates="owner")


class Intents(Base):
    __tablename__ = "intents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file = Column(String, index=True)
    description = Column(String, index=True)
    is_active = Column(Boolean, default=True)
    topic_id = Column(Integer, ForeignKey("topics.id"))

    topic = relationship("Topics", back_populates="intents")


class Questions(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    answer = Column(String, index=True)
    is_active = Column(Boolean, default=True)
    intent_id = Column(Integer, ForeignKey("intents.id"))

    intent = relationship("Intents", back_populates="questions")
