"""
ReAgent Main System Implementation

This module contains all agent implementations and the main system orchestrator.
It implements the complete multi-agent reversible reasoning framework.
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy
from enum import Enum
import hashlib
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
import openai
import numpy as np
from collections import defaultdict
import redis
import pickle
import time

from .models import Message, MessageType, KnowledgeAssertion, BacktrackingNode
from .monitoring import metrics_collector
from .config import settings, prompts

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== Core Components ==============

class LocalKnowledge:
    """
    Local knowledge management with backtracking support.
    
    Each agent maintains its own LocalKnowledge instance to track
    assertions and enable rollback to previous states.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.assertions: Dict[str, KnowledgeAssertion] = {}
        self.checkpoints: List[BacktrackingNode] = []
        self.current_state_id: Optional[str] = None
        self.max_checkpoints = settings.checkpoint_retention_hours * 12  # Every 5 min
        
    def add_assertion(self, assertion: KnowledgeAssertion) -> str:
        """Add new assertion and return its ID."""
        assertion_id = hashlib.md5(
            f"{assertion.content}_{datetime.now().isoformat()}_{self.agent_id}".encode()
        ).hexdigest()
        self.assertions[assertion_id] = assertion
        
        # Update metrics
        metrics_collector.update_knowledge_assertions(
            self.agent_id, len(self.assertions)
        )
        
        logger.debug(f"Agent {self.agent_id}: Added assertion {assertion_id[:8]}")
        return assertion_id
    
    def save_checkpoint(self) -> str:
        """Save current state as checkpoint."""
        state = {
            'assertions': deepcopy(self.assertions),
            'timestamp': datetime.now()
        }
        node = BacktrackingNode(
            state=state,
            timestamp=datetime.now(),
            parent_id=self.current_state_id
        )
        self.checkpoints.append(node)
        self.current_state_id = node.id
        
        # Cleanup old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
        
        # Update metrics
        metrics_collector.update_checkpoint_count(
            self.agent_id, len(self.checkpoints)
        )
        
        logger.info(f"Agent {self.agent_id}: Saved checkpoint {node.id[:8]}")
        return node.id
    
    def rollback(self, checkpoint_id: str) -> bool:
        """Rollback to specific checkpoint."""
        for checkpoint in self.checkpoints:
            if checkpoint.id == checkpoint_id:
                self.assertions = deepcopy(checkpoint.state['assertions'])
                self.current_state_id = checkpoint_id
                
                # Record backtracking event
                metrics_collector.record_backtracking(self.agent_id, "local")
                
                logger.info(f"Agent {self.agent_id}: Rolled back to {checkpoint_id[:8]}")
                return True
        
        logger.warning(f"Agent {self.agent_id}: Checkpoint {checkpoint_id[:8]} not found")
        return False
    
    def find_conflicts(self) -> List[Tuple[str, str]]:
        """Find conflicting assertions using various conflict detection strategies."""
        conflicts = []
        assertion_list = list(self.assertions.items())
        
        for i, (id1, assert1) in enumerate(assertion_list):
            for id2, assert2 in assertion_list[i+1:]:
                if self._are_conflicting(assert1, assert2):
                    conflicts.append((id1, id2))
        
        if conflicts:
            metrics_collector.record_conflict(self.agent_id)
            logger.warning(f"Agent {self.agent_id}: Found {len(conflicts)} conflicts")
            
        return conflicts
    
    def _are_conflicting(self, assert1: KnowledgeAssertion, assert2: KnowledgeAssertion) -> bool:
        """
        Check if two assertions conflict.
        
        This implements multiple conflict detection strategies:
        1. Direct negation detection
        2. Value conflict detection
        3. Logical inconsistency detection
        """
        content1 = assert1.content.lower()
        content2 = assert2.content.lower()
        
        # Strategy 1: Direct negation
        if self._check_direct_negation(content1, content2):
            return True
        
        # Strategy 2: Value conflicts
        if self._check_value_conflicts(content1, content2):
            return True
        
        # Strategy 3: Logical inconsistency
        if self._check_logical_inconsistency(content1, content2):
            return True
        
        return False
    
    def _check_direct_negation(self, content1: str, content2: str) -> bool:
        """Check for direct negation patterns."""
        negation_patterns = [
            ("not", ""),
            ("no", "yes"),
            ("false", "true"),
            ("incorrect", "correct"),
            ("wrong", "right"),
        ]
        
        for neg_word, pos_word in negation_patterns:
            if neg_word in content1 and content1.replace(neg_word, pos_word).strip() == content2.strip():
                return True
            if neg_word in content2 and content2.replace(neg_word, pos_word).strip() == content1.strip():
                return True
        
        return False
    
    def _check_value_conflicts(self, content1: str, content2: str) -> bool:
        """Check for conflicts in value assignments."""
        # Check for conflicting locations
        if all(term in content1 + content2 for term in ["capital", "city"]):
            cities1 = self._extract_cities(content1)
            cities2 = self._extract_cities(content2)
            if cities1 and cities2 and cities1 != cities2:
                return True
        
        # Check for conflicting numbers
        if "population" in content1 and "population" in content2:
            nums1 = self._extract_numbers(content1)
            nums2 = self._extract_numbers(content2)
            if nums1 and nums2:
                # If numbers differ by more than 10%, consider it a conflict
                for n1 in nums1:
                    for n2 in nums2:
                        if abs(n1 - n2) / max(n1, n2) > 0.1:
                            return True
        
        # Check for conflicting dates
        if any(term in content1 + content2 for term in ["year", "date", "when"]):
            dates1 = self._extract_years(content1)
            dates2 = self._extract_years(content2)
            if dates1 and dates2 and dates1 != dates2:
                return True
        
        return False
    
    def _check_logical_inconsistency(self, content1: str, content2: str) -> bool:
        """Check for logical inconsistencies."""
        # Mutual exclusivity patterns
        exclusive_pairs = [
            ("largest", "smallest"),
            ("highest", "lowest"),
            ("first", "last"),
            ("before", "after"),
            ("above", "below"),
            ("winner", "loser"),
        ]
        
        for word1, word2 in exclusive_pairs:
            if word1 in content1 and word2 in content2:
                # Check if they refer to the same entity
                if self._refer_to_same_entity(content1, content2):
                    return True
        
        return False
    
    def _extract_cities(self, text: str) -> Set[str]:
        """Extract city names from text."""
        # Simple implementation - in production, use NER
        common_cities = {
            "sacramento", "los angeles", "san francisco", "new york",
            "chicago", "boston", "seattle", "denver", "atlanta",
            "miami", "dallas", "phoenix", "philadelphia"
        }
        words = set(text.lower().split())
        return words.intersection(common_cities)
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text."""
        import re
        # Match numbers with optional commas and decimals
        pattern = r'[\d,]+\.?\d*'
        numbers = []
        for match in re.findall(pattern, text):
            try:
                # Remove commas and convert to float
                num = float(match.replace(',', ''))
                numbers.append(num)
            except ValueError:
                pass
        return numbers
    
    def _extract_years(self, text: str) -> Set[int]:
        """Extract year references from text."""
        import re
        # Match 4-digit years between 1900 and 2099
        pattern = r'\b(19\d{2}|20\d{2})\b'
        years = set()
        for match in re.findall(pattern, text):
            years.add(int(match))
        return years
    
    def _refer_to_same_entity(self, text1: str, text2: str) -> bool:
        """Check if two texts refer to the same entity."""
        # Simple implementation - check for common proper nouns
        words1 = set(word for word in text1.split() if word[0].isupper())
        words2 = set(word for word in text2.split() if word[0].isupper())
        common_words = words1.intersection(words2)
        return len(common_words) >= 2  # At least 2 common proper nouns

class MessageBus:
    """
    Message passing infrastructure using Redis.
    
    Implements pub/sub pattern for inter-agent communication
    with support for different message types and priorities.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=False,
            max_connections=settings.redis_max_connections
        )
        self.stream_key = "reagent:messages"
        self.subscribers: Dict[str, List[callable]] = defaultdict(list)
        self.message_queue_size = settings.message_queue_size
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
    async def publish(self, message: Message):
        """Publish message to stream with error handling and retries."""
        start_time = time.time()
        
        msg_data = {
            'type': message.type.value.encode(),
            'sender': message.sender.encode(),
            'content': json.dumps(message.content).encode(),
            'timestamp': message.timestamp.isoformat().encode(),
            'id': message.id.encode()
        }
        
        # Retry logic for transient failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add to stream with automatic trimming
                self.redis_client.xadd(
                    self.stream_key, 
                    msg_data,
                    maxlen=self.message_queue_size,
                    approximate=True
                )
                
                # Record metrics
                duration = time.time() - start_time
                metrics_collector.record_agent_message(message.sender, message.type.value)
                
                logger.debug(
                    f"Published message {message.id[:8]} from {message.sender} "
                    f"(type: {message.type.value}, latency: {duration:.3f}s)"
                )
                break
                
            except redis.RedisError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to publish message after {max_retries} attempts: {e}")
                    raise
        
    def subscribe(self, agent_id: str, handler: callable):
        """Subscribe agent to messages."""
        self.subscribers[agent_id].append(handler)
        logger.info(f"Agent {agent_id} subscribed to message bus")
        
    async def process_messages(self):
        """
        Process messages from stream.
        
        This runs continuously, reading messages from Redis stream
        and dispatching them to subscribed handlers.
        """
        last_id = '0'
        error_count = 0
        max_errors = 10
        
        while True:
            try:
                # Read messages with timeout
                messages = self.redis_client.xread(
                    {self.stream_key: last_id}, 
                    block=100,  # 100ms timeout
                    count=10    # Process up to 10 messages at once
                )
                
                for stream, stream_messages in messages:
                    for msg_id, data in stream_messages:
                        last_id = msg_id
                        
                        try:
                            # Decode message
                            message = Message(
                                type=MessageType(data[b'type'].decode()),
                                sender=data[b'sender'].decode(),
                                content=json.loads(data[b'content'].decode()),
                                timestamp=datetime.fromisoformat(
                                    data[b'timestamp'].decode()
                                ),
                                id=data[b'id'].decode()
                            )
                            
                            # Notify all subscribers
                            tasks = []
                            for agent_id, handlers in self.subscribers.items():
                                for handler in handlers:
                                    tasks.append(handler(message))
                            
                            # Execute handlers concurrently
                            if tasks:
                                await asyncio.gather(*tasks, return_exceptions=True)
                                
                        except Exception as e:
                            logger.error(f"Error processing message {msg_id}: {e}")
                            
                # Reset error count on successful iteration
                error_count = 0
                
            except redis.RedisError as e:
                error_count += 1
                logger.error(f"Redis error in message processing: {e}")
                
                if error_count >= max_errors:
                    logger.critical("Too many Redis errors, stopping message processing")
                    break
                    
                await asyncio.sleep(min(error_count, 5))  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error in message processing: {e}")
                await asyncio.sleep(1)

# ============== Base Agent Class ==============

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality for LLM interaction, message handling,
    and knowledge management.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        llm_client: openai.AsyncOpenAI,
        message_bus: MessageBus, 
        temperature: float = 0.6
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.message_bus = message_bus
        self.temperature = temperature
        self.local_knowledge = LocalKnowledge(agent_id)
        self.message_bus.subscribe(agent_id, self.handle_message)
        self.processing_times = []
        self.token_usage = {'input': 0, 'output': 0}
        
    @abstractmethod
    async def handle_message(self, message: Message):
        """Handle incoming message - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get agent-specific prompt template - must be implemented by subclasses."""
        pass
    
    async def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call LLM with error handling, retries, and metrics.
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retry attempts
            
        Returns:
            The LLM response content
            
        Raises:
            Exception: If all retry attempts fail
        """
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": self.get_prompt_template()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=settings.max_tokens_per_call,
                    response_format={"type": "json_object"},
                    timeout=settings.llm_request_timeout
                )
                
                # Record metrics
                duration = time.time() - start_time
                self.processing_times.append(duration)
                metrics_collector.record_agent_processing(self.agent_id, duration)
                
                # Estimate and record token usage
                content = response.choices[0].message.content
                usage = response.usage
                if usage:
                    metrics_collector.record_token_usage(
                        self.agent_id, 
                        usage.prompt_tokens, 
                        usage.completion_tokens
                    )
                    self.token_usage['input'] += usage.prompt_tokens
                    self.token_usage['output'] += usage.completion_tokens
                else:
                    # Fallback estimation
                    input_tokens = int(len(prompt.split()) * 1.3)
                    output_tokens = int(len(content.split()) * 1.3)
                    metrics_collector.record_token_usage(
                        self.agent_id, input_tokens, output_tokens
                    )
                
                logger.debug(
                    f"Agent {self.agent_id}: LLM call completed "
                    f"(latency: {duration:.3f}s, attempt: {attempt + 1})"
                )
                
                return content
                
            except openai.RateLimitError as e:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                logger.warning(
                    f"Rate limit hit for {self.agent_id}, waiting {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                
            except openai.APITimeoutError as e:
                logger.warning(
                    f"Timeout for {self.agent_id} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise
                    
            except Exception as e:
                logger.error(
                    f"LLM call failed for {self.agent_id} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent."""
        if not self.processing_times:
            return {
                'avg_processing_time': 0,
                'total_calls': 0,
                'total_tokens': 0
            }
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'total_calls': len(self.processing_times),
            'total_input_tokens': self.token_usage['input'],
            'total_output_tokens': self.token_usage['output'],
            'total_tokens': self.token_usage['input'] + self.token_usage['output']
        }

# ============== Execution Layer Agents ==============

class QuestionDecomposerAgent(BaseAgent):
    """
    Agent responsible for decomposing complex questions into sub-questions.
    
    This agent analyzes the input question and breaks it down into smaller,
    more manageable sub-questions that can be answered independently.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = prompts.get('question_decomposer', {}).get('temperature', 0.8)
        self.max_sub_questions = 5
        
    def get_prompt_template(self) -> str:
        """Get the prompt template for question decomposition."""
        return prompts.get('question_decomposer', {}).get('system_prompt', '''
You are the Question-Decomposer Agent, specializing in breaking down complex queries into manageable sub-questions.

Your Goals:
1. Parse the original query into logically independent or sequential sub-questions
2. Preserve all necessary context
3. Output in structured JSON format

Guidelines:
- Each sub-question should be self-contained but may depend on previous answers
- Order sub-questions logically (dependencies first)
- Include temporal or causal relationships
- Maximum 5 sub-questions

Output Format (JSON Only):
{
  "sub_questions": ["Sub-question 1", "Sub-question 2", ...],
  "decomposition_reasoning": "Short explanation of decomposition process",
  "dependencies": {"q2": ["q1"], "q3": ["q1", "q2"]}  // Optional
}
''')
    
    async def handle_message(self, message: Message):
        """Handle incoming messages for question decomposition."""
        if message.type == MessageType.ASSERT and "original_question" in message.content:
            await self.decompose_question(message.content["original_question"])
    
    async def decompose_question(self, question: str):
        """
        Decompose complex question into sub-questions.
        
        Args:
            question: The original complex question to decompose
        """
        checkpoint_id = self.local_knowledge.save_checkpoint()
        
        try:
            # Analyze question complexity
            complexity = self._analyze_complexity(question)
            
            # Adjust prompt based on complexity
            enhanced_prompt = f"""Decompose this {'highly ' if complexity > 0.7 else ''}complex question: {question}

Consider:
- Temporal dependencies (what happened first?)
- Causal relationships (what caused what?)
- Entity relationships (who/what is connected?)
- Comparison requirements (what needs to be compared?)
- Final synthesis needed

Complexity score: {complexity:.2f}"""
            
            response = await self.call_llm(enhanced_prompt)
            result = json.loads(response)
            
            # Validate decomposition
            if not self._validate_decomposition(result, question):
                raise ValueError("Invalid decomposition result")
            
            # Store decomposition in local knowledge
            decomposition_id = self.local_knowledge.add_assertion(
                KnowledgeAssertion(
                    content=f"decomposition: {json.dumps(result)}",
                    source=self.agent_id,
                    confidence=0.9
                )
            )
            
            # Store each sub-question
            for i, sub_q in enumerate(result["sub_questions"][:self.max_sub_questions]):
                assertion = KnowledgeAssertion(
                    content=f"sub_question_{i}: {sub_q}",
                    source=self.agent_id,
                    confidence=0.9,
                    dependencies=[decomposition_id] if i > 0 else []
                )
                self.local_knowledge.add_assertion(assertion)
            
            # Broadcast sub-questions
            await self.message_bus.publish(Message(
                type=MessageType.INFORM,
                sender=self.agent_id,
                content={
                    "sub_questions": result["sub_questions"][:self.max_sub_questions],
                    "reasoning": result["decomposition_reasoning"],
                    "dependencies": result.get("dependencies", {}),
                    "complexity": complexity
                }
            ))
            
            logger.info(
                f"Decomposed question into {len(result['sub_questions'])} "
                f"sub-questions (complexity: {complexity:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            self.local_knowledge.rollback(checkpoint_id)
            
            await self.message_bus.publish(Message(
                type=MessageType.REJECT,
                sender=self.agent_id,
                content={
                    "error": str(e),
                    "original_question": question
                }
            ))
    
    def _analyze_complexity(self, question: str) -> float:
        """
        Analyze question complexity to guide decomposition.
        
        Returns a score between 0 and 1 indicating complexity.
        """
        complexity_indicators = {
            # Temporal indicators
            'when': 0.2, 'before': 0.3, 'after': 0.3, 'during': 0.2,
            'while': 0.2, 'since': 0.2, 'until': 0.2,
            
            # Comparison indicators
            'compare': 0.4, 'difference': 0.3, 'similar': 0.3,
            'larger': 0.2, 'smaller': 0.2, 'more': 0.2, 'less': 0.2,
            
            # Causal indicators
            'because': 0.3, 'cause': 0.3, 'result': 0.3, 'therefore': 0.3,
            'consequence': 0.3, 'lead to': 0.3, 'due to': 0.3,
            
            # Multi-entity indicators
            'both': 0.2, 'either': 0.2, 'neither': 0.2, 'all': 0.2,
            'multiple': 0.3, 'several': 0.2, 'various': 0.2,
            
            # Conditional indicators
            'if': 0.3, 'given': 0.3, 'provided': 0.3, 'assuming': 0.3,
            'unless': 0.3, 'except': 0.2, 'but': 0.2
        }
        
        question_lower = question.lower()
        score = 0.1  # Base complexity
        
        for indicator, weight in complexity_indicators.items():
            if indicator in question_lower:
                score += weight
        
        # Length factor
        word_count = len(question.split())
        if word_count > 50:
            score += 0.2
        elif word_count > 30:
            score += 0.1
        
        # Punctuation complexity
        if question.count(',') > 2:
            score += 0.1
        if ';' in question or ':' in question:
            score += 0.1
        
        return min(score, 1.0)
    
    def _validate_decomposition(self, result: Dict[str, Any], original: str) -> bool:
        """Validate the decomposition result."""
        if not isinstance(result, dict):
            return False
        
        if "sub_questions" not in result:
            return False
        
        sub_questions = result.get("sub_questions", [])
        
        # Check if we have valid sub-questions
        if not sub_questions or not isinstance(sub_questions, list):
            return False
        
        # Each sub-question should be a non-empty string
        for sq in sub_questions:
            if not isinstance(sq, str) or not sq.strip():
                return False
        
        # Should have reasoning
        if not result.get("decomposition_reasoning"):
            return False
        
        return True

class RetrieverAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant evidence.
    
    This agent searches various knowledge sources (documents, databases,
    knowledge graphs) to find evidence relevant to the sub-questions.
    """
    
    def __init__(self, *args, retrieval_backend=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieval_backend = retrieval_backend
        self.temperature = prompts.get('retriever', {}).get('temperature', 0.6)
        self.max_evidence_per_question = 5
        
    def get_prompt_template(self) -> str:
        """Get the prompt template for evidence retrieval."""
        return prompts.get('retriever', {}).get('system_prompt', '''
You are the Retriever Agent, responsible for fetching relevant evidence.

Your Goals:
1. Retrieve the most relevant facts or passages
2. Include confidence scores
3. Return findings in JSON structure

Guidelines:
- Focus on factual, verifiable information
- Prefer authoritative sources
- Include context for each piece of evidence
- Rate confidence based on source reliability and relevance

Output Format (JSON Only):
{
  "retrieved_evidence": [
    {
      "source": "source name",
      "content": "relevant content", 
      "confidence": 0.0-1.0,
      "relevance": "how this answers the question",
      "metadata": {"year": 2024, "type": "article"}  // Optional
    }
  ],
  "retrieval_reasoning": "Short justification"
}
''')
    
    async def handle_message(self, message: Message):
        """Handle incoming messages for evidence retrieval."""
        if message.type == MessageType.INFORM and "sub_questions" in message.content:
            sub_questions = message.content["sub_questions"]
            dependencies = message.content.get("dependencies", {})
            
            # Process sub-questions considering dependencies
            for i, sub_q in enumerate(sub_questions):
                # Check if dependencies are satisfied
                deps = dependencies.get(f"q{i+1}", [])
                if self._dependencies_satisfied(deps):
                    await self.retrieve_evidence(sub_q, context=message.content)
    
    async def retrieve_evidence(self, sub_question: str, context: Dict[str, Any] = None):
        """
        Retrieve evidence for a sub-question.
        
        Args:
            sub_question: The sub-question to find evidence for
            context: Additional context from the decomposition
        """
        checkpoint_id = self.local_knowledge.save_checkpoint()
        
        try:
            # Enhance query with context
            enhanced_query = self._enhance_query(sub_question, context)
            
            # If we have a retrieval backend, use it
            if self.retrieval_backend:
                evidence_list = await self._retrieve_from_backend(enhanced_query)
            else:
                # Otherwise, use LLM to simulate retrieval
                evidence_list = await self._retrieve_with_llm(sub_question, enhanced_query)
            
            # Filter and rank evidence
            filtered_evidence = self._filter_evidence(evidence_list, sub_question)
            
            # Store retrieved evidence
            for evidence in filtered_evidence[:self.max_evidence_per_question]:
                assertion = KnowledgeAssertion(
                    content=evidence["content"],
                    source=evidence["source"],
                    confidence=evidence["confidence"]
                )
                self.local_knowledge.add_assertion(assertion)
            
            # Broadcast retrieved evidence
            await self.message_bus.publish(Message(
                type=MessageType.INFORM,
                sender=self.agent_id,
                content={
                    "sub_question": sub_question,
                    "evidence": filtered_evidence[:self.max_evidence_per_question],
                    "reasoning": f"Retrieved {len(filtered_evidence)} relevant pieces of evidence",
                    "total_found": len(evidence_list)
                }
            ))
            
            logger.info(
                f"Retrieved {len(filtered_evidence)} evidence pieces for: "
                f"{sub_question[:50]}..."
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            self.local_knowledge.rollback(checkpoint_id)
            
            await self.message_bus.publish(Message(
                type=MessageType.REJECT,
                sender=self.agent_id,
                content={
                    "error": str(e),
                    "sub_question": sub_question
                }
            ))
    
    def _enhance_query(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance query with additional context for better retrieval."""
        enhanced = query
        
        # Add temporal context if present
        if "temporal_context" in context:
            enhanced += f" (Time period: {context['temporal_context']})"
        
        # Add entity context if present
        if "entities" in context:
            enhanced += f" (Related to: {', '.join(context['entities'])})"
        
        # Add complexity hint
        if context and context.get("complexity", 0) > 0.5:
            enhanced += " (Requires multiple sources)"
        
        return enhanced
    
    async def _retrieve_from_backend(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve from actual backend (Elasticsearch, vector DB, etc.)."""
        # This would connect to your actual retrieval system
        # For now, returning empty list
        return []
    
    async def _retrieve_with_llm(self, sub_question: str, enhanced_query: str) -> List[Dict[str, Any]]:
        """Use LLM to simulate retrieval when no backend is available."""
        prompt = f"""Retrieve evidence for: {sub_question}

Enhanced query: {enhanced_query}

Provide factual information that would help answer this question.
Focus on specific facts, dates, numbers, and relationships.
Include source attribution for credibility."""
        
        response = await self.call_llm(prompt)
        result = json.loads(response)
        
        return result.get("retrieved_evidence", [])
    
    def _filter_evidence(self, evidence_list: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """Filter and rank evidence by relevance and confidence."""
        # Score each piece of evidence
        scored_evidence = []
        
        for evidence in evidence_list:
            score = self._score_evidence(evidence, question)
            evidence['relevance_score'] = score
            scored_evidence.append(evidence)
        
        # Sort by combined score (confidence * relevance)
        scored_evidence.sort(
            key=lambda x: x.get('confidence', 0.5) * x.get('relevance_score', 0.5),
            reverse=True
        )
        
        return scored_evidence
    
    def _score_evidence(self, evidence: Dict[str, Any], question: str) -> float:
        """Score evidence relevance to the question."""
        content = evidence.get('content', '').lower()
        question_lower = question.lower()
        
        # Extract keywords from question
        question_words = set(
            word for word in question_lower.split()
            if len(word) > 3 and word not in {'what', 'when', 'where', 'which', 'who', 'how'}
        )
        
        # Count keyword matches
        matches = sum(1 for word in question_words if word in content)
        
        # Calculate relevance score
        if not question_words:
            return 0.5
        
        relevance = matches / len(question_words)
        
        # Boost for exact phrase matches
        if any(phrase in content for phrase in self._extract_phrases(question_lower)):
            relevance *= 1.5
        
        return min(relevance, 1.0)
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract important phrases from text."""
        # Simple bigram extraction
        words = text.split()
        phrases = []
        
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:
                phrases.append(f"{words[i]} {words[i + 1]}")
        
        return phrases
    
    def _dependencies_satisfied(self, dependencies: List[str]) -> bool:
        """Check if required dependencies are satisfied."""
        # For now, assume dependencies are satisfied
        # In production, check against completed sub-questions
        return True

class VerifierAgent(BaseAgent):
    """
    Agent responsible for verifying consistency and triggering local backtracking.
    
    This agent checks new evidence against existing knowledge, detects conflicts,
    and initiates backtracking when inconsistencies are found.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = prompts.get('verifier', {}).get('temperature', 0.6)
        self.conflict_threshold = 0.7  # Confidence threshold for conflict detection
        
    def get_prompt_template(self) -> str:
        """Get the prompt template for verification."""
        return prompts.get('verifier', {}).get('system_prompt', '''
You are the Verifier Agent, focusing on consistency and correctness.

Your Goals:
1. Validate new information against existing knowledge
2. Identify contradictions
3. Produce verified facts or signal conflicts

Guidelines:
- Check for logical consistency
- Identify conflicting values (numbers, dates, names)
- Verify causal relationships
- Flag uncertain or ambiguous information

Output Format (JSON Only):
{
  "verified_facts": ["Fact 1", "Fact 2", ...],
  "conflicts_detected": [
    {
      "description": "Nature of conflict",
      "conflicting_items": ["Item 1", "Item 2"],
      "confidence": 0.0-1.0
    }
  ],
  "local_backtracking_action": "Description or 'none'",
  "verification_notes": "Additional observations"
}
''')
    
    async def handle_message(self, message: Message):
        """Handle incoming messages for verification."""
        if message.type == MessageType.INFORM and "evidence" in message.content:
            await self.verify_evidence(message.content)
        elif message.type == MessageType.ASSERT and "facts" in message.content:
            await self.verify_facts(message.content["facts"])
    
    async def verify_evidence(self, content: Dict[str, Any]):
        """
        Verify evidence and handle local conflicts.
        
        Args:
            content: Dictionary containing evidence to verify
        """
        checkpoint_id = self.local_knowledge.save_checkpoint()
        
        try:
            # Add new evidence to local knowledge temporarily
            temp_assertions = []
            for evidence in content.get("evidence", []):
                assertion = KnowledgeAssertion(
                    content=evidence["content"],
                    source=evidence["source"],
                    confidence=evidence["confidence"]
                )
                assertion_id = self.local_knowledge.add_assertion(assertion)
                temp_assertions.append(assertion_id)
            
            # Check for conflicts
            conflicts = self.local_knowledge.find_conflicts()
            
            if conflicts:
                logger.info(f"Verifier detected {len(conflicts)} conflicts")
                
                # Analyze conflicts with LLM
                conflict_analysis = await self._analyze_conflicts(conflicts)
                
                # Decide on backtracking
                if self._should_backtrack(conflict_analysis):
                    # Perform local backtracking
                    self.local_knowledge.rollback(checkpoint_id)
                    
                    # Broadcast conflict and backtracking
                    await self.message_bus.publish(Message(
                        type=MessageType.CHALLENGE,
                        sender=self.agent_id,
                        content={
                            "conflicts": conflict_analysis["conflicts_detected"],
                            "backtracking_action": conflict_analysis["local_backtracking_action"],
                            "verified_facts": [],
                            "sub_question": content.get("sub_question", ""),
                            "rejected_evidence": content.get("evidence", [])
                        }
                    ))
                else:
                    # Conflicts exist but are manageable
                    await self._handle_minor_conflicts(conflict_analysis, content)
            else:
                # No conflicts, verify and broadcast facts
                verified_facts = await self._verify_facts(content.get("evidence", []))
                
                await self.message_bus.publish(Message(
                    type=MessageType.ASSERT,
                    sender=self.agent_id,
                    content={
                        "verified_facts": verified_facts,
                        "sub_question": content.get("sub_question", ""),
                        "confidence": self._calculate_overall_confidence(verified_facts)
                    }
                ))
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            self.local_knowledge.rollback(checkpoint_id)
            
            await self.message_bus.publish(Message(
                type=MessageType.REJECT,
                sender=self.agent_id,
                content={
                    "error": str(e),
                    "evidence": content.get("evidence", [])
                }
            ))
    
    async def verify_facts(self, facts: List[str]):
        """Verify a list of facts for consistency."""
        # Similar to verify_evidence but for already processed facts
        pass
    
    async def _analyze_conflicts(self, conflicts: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze detected conflicts using LLM."""
        # Prepare conflict descriptions
        conflict_descriptions = []
        
        for id1, id2 in conflicts[:5]:  # Limit to 5 conflicts for LLM
            assert1 = self.local_knowledge.assertions[id1]
            assert2 = self.local_knowledge.assertions[id2]
            
            conflict_descriptions.append(
                f"Conflict between:\n"
                f"1. {assert1.content} (confidence: {assert1.confidence})\n"
                f"2. {assert2.content} (confidence: {assert2.confidence})"
            )
        
        prompt = f"""Analyze these conflicts and suggest resolution:

{chr(10).join(conflict_descriptions)}

For each conflict:
1. Determine the nature and severity
2. Suggest which assertion to keep (if any)
3. Recommend if backtracking is needed"""
        
        response = await self.call_llm(prompt)
        return json.loads(response)
    
    def _should_backtrack(self, conflict_analysis: Dict[str, Any]) -> bool:
        """Determine if backtracking is necessary based on conflict analysis."""
        if conflict_analysis.get("local_backtracking_action") == "none":
            return False
        
        # Check conflict severity
        conflicts = conflict_analysis.get("conflicts_detected", [])
        if not conflicts:
            return False
        
        # Backtrack if any conflict has high confidence
        for conflict in conflicts:
            if conflict.get("confidence", 0) > self.conflict_threshold:
                return True
        
        # Backtrack if multiple medium-confidence conflicts
        medium_conflicts = sum(
            1 for c in conflicts 
            if 0.5 <= c.get("confidence", 0) <= self.conflict_threshold
        )
        
        return medium_conflicts >= 3
    
    async def _handle_minor_conflicts(self, conflict_analysis: Dict[str, Any], content: Dict[str, Any]):
        """Handle conflicts that don't require full backtracking."""
        # Remove only the conflicting assertions
        conflicts_to_remove = set()
        
        for conflict in conflict_analysis.get("conflicts_detected", []):
            items = conflict.get("conflicting_items", [])
            # Remove the lower confidence item
            if len(items) >= 2:
                # Find the assertion with lower confidence
                assertions = [
                    (aid, self.local_knowledge.assertions.get(aid))
                    for aid in self.local_knowledge.assertions
                    if any(item in self.local_knowledge.assertions.get(aid, KnowledgeAssertion("", "", 0)).content 
                          for item in items)
                ]
                
                if assertions:
                    # Sort by confidence and remove lowest
                    assertions.sort(key=lambda x: x[1].confidence if x[1] else 0)
                    if assertions[0][0] in self.local_knowledge.assertions:
                        del self.local_knowledge.assertions[assertions[0][0]]
        
        # Verify remaining facts
        remaining_evidence = [
            e for e in content.get("evidence", [])
            if not any(
                item in e.get("content", "")
                for conflict in conflict_analysis.get("conflicts_detected", [])
                for item in conflict.get("conflicting_items", [])
            )
        ]
        
        verified_facts = await self._verify_facts(remaining_evidence)
        
        # Broadcast results with conflict warning
        await self.message_bus.publish(Message(
            type=MessageType.ASSERT,
            sender=self.agent_id,
            content={
                "verified_facts": verified_facts,
                "conflicts_resolved": conflict_analysis.get("conflicts_detected", []),
                "sub_question": content.get("sub_question", ""),
                "confidence": self._calculate_overall_confidence(verified_facts) * 0.8  # Reduce confidence due to conflicts
            }
        ))
    
    async def _verify_facts(self, evidence_list: List[Dict[str, Any]]) -> List[str]:
        """Verify and extract facts from evidence."""
        if not evidence_list:
            return []
        
        # Use LLM to extract and verify facts
        evidence_text = "\n".join([
            f"- {e['content']} (source: {e['source']}, confidence: {e['confidence']})"
            for e in evidence_list
        ])
        
        prompt = f"""Extract and verify facts from this evidence:

{evidence_text}

Return only facts that are:
1. Clearly stated (not inferred)
2. Specific and verifiable
3. Relevant to the question
4. Not contradictory"""
        
        response = await self.call_llm(prompt)
        result = json.loads(response)
        
        return result.get("verified_facts", [])
    
    def _calculate_overall_confidence(self, facts: List[Any]) -> float:
        """Calculate overall confidence score for verified facts."""
        if not facts:
            return 0.0
        
        # Base confidence on number and quality of facts
        base_confidence = min(len(facts) * 0.2, 0.8)
        
        # Adjust based on source diversity
        sources = set()
        for fact in facts:
            if isinstance(fact, dict) and "source" in fact:
                sources.add(fact["source"])
        
        diversity_bonus = min(len(sources) * 0.1, 0.2)
        
        return min(base_confidence + diversity_bonus, 1.0)

class AnswerAssemblerAgent(BaseAgent):
    """
    Agent responsible for assembling final answer from partial results.
    
    This agent synthesizes verified facts into a coherent answer,
    handling any remaining conflicts and formatting the response.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_answers = {}
        self.verified_facts = defaultdict(list)
        self.min_facts_to_assemble = 3
        self.assembly_timeout = 60  # seconds
        self.assembly_start_time = None
        
    def get_prompt_template(self) -> str:
        """Get the prompt template for answer assembly."""
        return prompts.get('answer_assembler', {}).get('system_prompt', '''
You are the Answer-Assembler Agent, synthesizing coherent responses.

Your Goals:
1. Aggregate partial answers logically
2. Compose final answer
3. Escalate unresolvable contradictions

Guidelines:
- Synthesize facts into a natural, complete answer
- Maintain logical flow and coherence
- Highlight any uncertainties or gaps
- Format for clarity and readability

Output Format (JSON Only):
{
  "final_answer": "Complete answer to the query",
  "partial_answer_synthesis": ["How partial answers were combined"],
  "confidence_score": 0.0-1.0,
  "supporting_facts": ["Key facts used"],
  "escalation_signal": "none" or reason for escalation,
  "answer_metadata": {
    "answer_type": "factual|comparative|explanatory|list",
    "certainty": "high|medium|low"
  }
}
''')
    
    async def handle_message(self, message: Message):
        """Handle incoming messages for answer assembly."""
        if message.type == MessageType.ASSERT and "verified_facts" in message.content:
            sub_question = message.content.get("sub_question", "general")
            facts = message.content["verified_facts"]
            confidence = message.content.get("confidence", 0.8)
            
            # Store verified facts
            self.verified_facts[sub_question].extend(facts)
            self.partial_answers[message.sender] = {
                "facts": facts,
                "confidence": confidence,
                "timestamp": datetime.now()
            }
            
            # Start assembly timer if not started
            if self.assembly_start_time is None:
                self.assembly_start_time = time.time()
            
            # Check if ready to assemble
            await self.check_assembly_ready()
            
        elif message.type == MessageType.CHALLENGE:
            # Handle conflicts that might affect assembly
            await self.handle_conflict(message.content)
    
    async def check_assembly_ready(self):
        """Check if we have enough partial answers to assemble."""
        total_facts = sum(len(facts) for facts in self.verified_facts.values())
        
        # Assembly conditions
        ready = False
        reason = ""
        
        if total_facts >= self.min_facts_to_assemble:
            ready = True
            reason = "Sufficient facts collected"
        elif self.assembly_start_time and (time.time() - self.assembly_start_time) > self.assembly_timeout:
            ready = True
            reason = "Assembly timeout reached"
        elif len(self.partial_answers) >= 3:  # Enough agents have responded
            ready = True
            reason = "Sufficient agent responses"
        
        if ready:
            logger.info(f"Starting answer assembly: {reason}")
            await self.assemble_answer()
    
    async def assemble_answer(self):
        """Assemble final answer from partial results."""
        checkpoint_id = self.local_knowledge.save_checkpoint()
        
        try:
            # Gather all verified facts
            all_facts = []
            fact_sources = defaultdict(list)
            
            for sub_q, facts in self.verified_facts.items():
                for fact in facts:
                    all_facts.append(fact)
                    fact_sources[fact].append(sub_q)
            
            if not all_facts:
                raise ValueError("No facts available for assembly")
            
            # Check for remaining conflicts
            conflicts = self._detect_fact_conflicts(all_facts)
            
            if conflicts and len(conflicts) > 2:
                # Too many conflicts, escalate to supervisor
                await self._escalate_conflicts(conflicts, all_facts)
                return
            
            # Prepare assembly prompt
            assembly_prompt = self._prepare_assembly_prompt(all_facts, fact_sources)
            
            # Generate final answer
            response = await self.call_llm(assembly_prompt)
            result = json.loads(response)
            
            # Validate assembly result
            if not self._validate_assembly(result):
                raise ValueError("Invalid assembly result")
            
            # Store final answer
            self.local_knowledge.add_assertion(
                KnowledgeAssertion(
                    content=f"final_answer: {result['final_answer']}",
                    source=self.agent_id,
                    confidence=result.get("confidence_score", 0.8)
                )
            )
            
            # Broadcast final answer
            await self.message_bus.publish(Message(
                type=MessageType.INFORM,
                sender=self.agent_id,
                content={
                    "final_answer": result["final_answer"],
                    "synthesis": result["partial_answer_synthesis"],
                    "confidence": result.get("confidence_score", 0.8),
                    "supporting_facts": result.get("supporting_facts", []),
                    "metadata": result.get("answer_metadata", {}),
                    "assembly_time": time.time() - (self.assembly_start_time or time.time())
                }
            ))
            
            logger.info(
                f"Assembled final answer with confidence: "
                f"{result.get('confidence_score', 0.8):.2f}"
            )
            
            # Reset for next question
            self._reset_assembly_state()
            
        except Exception as e:
            logger.error(f"Answer assembly failed: {e}")
            self.local_knowledge.rollback(checkpoint_id)
            
            # Check if we should escalate
            if "conflict" in str(e).lower():
                await self._escalate_conflicts([], all_facts)
            else:
                await self.message_bus.publish(Message(
                    type=MessageType.REJECT,
                    sender=self.agent_id,
                    content={
                        "error": str(e),
                        "partial_results": dict(self.verified_facts)
                    }
                ))
    
    async def handle_conflict(self, conflict_info: Dict[str, Any]):
        """Handle conflict information from other agents."""
        # Remove conflicting facts from our collection
        rejected_facts = conflict_info.get("rejected_evidence", [])
        
        for sub_q, facts in self.verified_facts.items():
            self.verified_facts[sub_q] = [
                f for f in facts 
                if not any(
                    rej in str(f) 
                    for rej in rejected_facts
                )
            ]
    
    def _detect_fact_conflicts(self, facts: List[str]) -> List[Dict[str, Any]]:
        """Detect conflicts among facts."""
        conflicts = []
        
        # Simple conflict detection based on contradictory patterns
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                if self._facts_conflict(fact1, fact2):
                    conflicts.append({
                        "fact1": fact1,
                        "fact2": fact2,
                        "type": "contradiction"
                    })
        
        return conflicts
    
    def _facts_conflict(self, fact1: str, fact2: str) -> bool:
        """Check if two facts conflict."""
        # Simple heuristics
        fact1_lower = fact1.lower()
        fact2_lower = fact2.lower()
        
        # Check for opposite statements
        opposites = [
            ("is", "is not"),
            ("was", "was not"),
            ("has", "does not have"),
            ("larger", "smaller"),
            ("before", "after"),
            ("true", "false")
        ]
        
        for pos, neg in opposites:
            if pos in fact1_lower and neg in fact2_lower:
                return True
            if neg in fact1_lower and pos in fact2_lower:
                return True
        
        return False
    
    async def _escalate_conflicts(self, conflicts: List[Dict[str, Any]], facts: List[str]):
        """Escalate conflicts to supervisor."""
        await self.message_bus.publish(Message(
            type=MessageType.CHALLENGE,
            sender=self.agent_id,
            content={
                "reason": f"Unable to resolve {len(conflicts)} conflicts during assembly",
                "conflicts": conflicts,
                "partial_answers": dict(self.partial_answers),
                "all_facts": facts
            }
        ))
        
        logger.warning(f"Escalated {len(conflicts)} conflicts to supervisor")
    
    def _prepare_assembly_prompt(self, facts: List[str], sources: Dict[str, List[str]]) -> str:
        """Prepare prompt for final answer assembly."""
        # Organize facts by sub-question
        organized_facts = defaultdict(list)
        for fact, sub_questions in sources.items():
            for sq in sub_questions:
                organized_facts[sq].append(fact)
        
        fact_text = ""
        for sq, sq_facts in organized_facts.items():
            if sq != "general":
                fact_text += f"\nRegarding '{sq}':\n"
            fact_text += "\n".join(f"- {fact}" for fact in sq_facts)
        
        return f"""Synthesize these verified facts into a complete, coherent answer:

{fact_text}

Requirements:
1. Answer the original question completely
2. Use all relevant facts
3. Maintain logical flow
4. Be concise but comprehensive
5. Indicate any uncertainties"""
    
    def _validate_assembly(self, result: Dict[str, Any]) -> bool:
        """Validate the assembly result."""
        required_fields = ["final_answer", "partial_answer_synthesis"]
        
        for field in required_fields:
            if field not in result:
                return False
        
        # Check answer quality
        answer = result.get("final_answer", "")
        if not answer or len(answer) < 10:
            return False
        
        # Check confidence score
        confidence = result.get("confidence_score", 0)
        if not 0 <= confidence <= 1:
            return False
        
        return True
    
    def _reset_assembly_state(self):
        """Reset assembly state for next question."""
        self.partial_answers.clear()
        self.verified_facts.clear()
        self.assembly_start_time = None

# ============== Supervisor Layer Agents ==============

class SupervisorAgent(BaseAgent):
    """
    Agent responsible for global conflict resolution and backtracking.
    
    This agent handles system-wide inconsistencies that cannot be resolved
    locally, coordinating rollback across multiple agents when necessary.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_knowledge = {}  # Track all agents' knowledge
        self.global_checkpoints = []
        self.conflict_history = []
        self.max_global_checkpoints = 10
        
    def get_prompt_template(self) -> str:
        """Get the prompt template for supervision."""
        return prompts.get('supervisor', {}).get('system_prompt', '''
You are the Supervisor Agent, orchestrating global conflict resolution.

Your Goals:
1. Collect escalation signals
2. Execute system-wide rollback if needed
3. Provide summary of changes

                    "action": "local_rollback",
                    "target_timestamp": checkpoint["timestamp"],
                    "reason": "Global rollback initiated"
                }
            ))
        
        logger.info(
            f"Global rollback to checkpoint {checkpoint_id[:8]} completed, "
            f"affecting {len(affected_agents)} agents"
        )
        return True
    
    async def _execute_partial_resolution(
        self,
        analysis: Dict[str, Any],
        strategy: Dict[str, Any]
    ):
        """Execute partial resolution without full rollback."""
        # Identify specific facts to remove or modify
        facts_to_remove = []
        facts_to_modify = []
        
        for conflict in analysis.get("conflict_summary", []):
            if isinstance(conflict, str):
                # Simple conflict description
                facts_to_remove.append(conflict)
            elif isinstance(conflict, dict):
                # Detailed conflict with resolution
                if conflict.get("resolution") == "remove":
                    facts_to_remove.extend(conflict.get("facts", []))
                elif conflict.get("resolution") == "modify":
                    facts_to_modify.extend(conflict.get("facts", []))
        
        # Broadcast partial resolution command
        await self.message_bus.publish(Message(
            type=MessageType.INFORM,
            sender=self.agent_id,
            content={
                "action": "partial_resolution",
                "remove_facts": facts_to_remove,
                "modify_facts": facts_to_modify,
                "new_consensus": analysis["updated_consensus_state"],
                "reasoning": analysis["reasoning_notes"]
            }
        ))
        
        logger.info(
            f"Partial resolution executed: removing {len(facts_to_remove)} facts, "
            f"modifying {len(facts_to_modify)} facts"
        )
    
    async def _handle_rollback_failure(self, analysis: Dict[str, Any]):
        """Handle cases where rollback fails."""
        logger.error("Global rollback failed, attempting recovery")
        
        # Try to establish minimal consistent state
        minimal_facts = self._extract_minimal_consistent_facts()
        
        # Broadcast recovery attempt
        await self.message_bus.publish(Message(
            type=MessageType.CHALLENGE,
            sender=self.agent_id,
            content={
                "action": "recovery",
                "minimal_facts": minimal_facts,
                "reasoning": "Rollback failed, establishing minimal consistent state"
            }
        ))
    
    async def _handle_critical_failure(self, error: Exception, original_message: Message):
        """Handle critical failures in the supervisor."""
        logger.critical(f"Supervisor critical failure: {error}")
        
        # Broadcast emergency shutdown
        await self.message_bus.publish(Message(
            type=MessageType.REJECT,
            sender=self.agent_id,
            content={
                "action": "emergency_shutdown",
                "error": str(error),
                "original_conflict": original_message.content,
                "recommendation": "Manual intervention required"
            }
        ))
    
    def _extract_minimal_consistent_facts(self) -> List[str]:
        """Extract minimal set of facts that are consistent."""
        # Start with consensus facts
        facts = self._extract_consensus_facts()
        
        # Remove any facts that appear in conflicts
        conflicted_facts = set()
        for conflict in self.conflict_history:
            if "content" in conflict and "conflicts" in conflict["content"]:
                for c in conflict["content"]["conflicts"]:
                    if isinstance(c, dict):
                        conflicted_facts.update(c.get("facts", []))
        
        minimal_facts = [f for f in facts if f not in conflicted_facts]
        
        return minimal_facts

class ControllerAgent(BaseAgent):
    """
    Agent providing high-level strategic oversight.
    
    This agent monitors system behavior, intervenes when patterns indicate
    problems, and can override decisions when necessary.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_reliability = defaultdict(lambda: 1.0)
        self.intervention_history = []
        self.performance_metrics = defaultdict(lambda: {"successes": 0, "failures": 0})
        self.intervention_threshold = 3  # Number of issues before intervention
        
    def get_prompt_template(self) -> str:
        """Get the prompt template for controller."""
        return prompts.get('controller', {}).get('system_prompt', '''
You are the Controller Agent, providing strategic oversight.

Your Goals:
1. Intervene when standard backtracking fails
2. Challenge critical assumptions
3. Log meta-data about reliability

Guidelines:
- Monitor system-wide patterns
- Identify systemic issues
- Override when necessary for system health
- Maintain agent performance metrics

Output Format (JSON Only):
{
  "intervention_type": "challenge|override|escalate|none",
  "target_of_intervention": "Agent or assertion to target",
  "rationale": "Why intervention is needed",
  "meta_notes": "Additional commentary",
  "system_health": {
    "status": "healthy|degraded|critical",
    "issues": ["List of identified issues"],
    "recommendations": ["Suggested actions"]
  }
}
''')
    
    async def handle_message(self, message: Message):
        """Monitor messages for intervention opportunities."""
        # Update performance metrics
        self._update_performance_metrics(message)
        
        # Monitor for issues requiring intervention
        if message.type == MessageType.CHALLENGE:
            self.intervention_history.append({
                "timestamp": datetime.now(),
                "sender": message.sender,
                "content": message.content,
                "type": "conflict"
            })
            
            if self.should_intervene():
                await self.strategic_intervention()
                
        elif message.type == MessageType.REJECT:
            self.intervention_history.append({
                "timestamp": datetime.now(),
                "sender": message.sender,
                "content": message.content,
                "type": "failure"
            })
            
            # Update reliability for failed agent
            self.agent_reliability[message.sender] *= 0.9
            
        elif message.type == MessageType.INFORM and "final_answer" in message.content:
            # Success case
            self.performance_metrics[message.sender]["successes"] += 1
    
    def _update_performance_metrics(self, message: Message):
        """Update performance metrics based on message."""
        sender = message.sender
        
        if message.type == MessageType.ASSERT:
            self.performance_metrics[sender]["assertions"] = \
                self.performance_metrics[sender].get("assertions", 0) + 1
                
        elif message.type == MessageType.CHALLENGE:
            self.performance_metrics[sender]["conflicts"] = \
                self.performance_metrics[sender].get("conflicts", 0) + 1
            self.agent_reliability[sender] *= 0.95
            
        elif message.type == MessageType.REJECT:
            self.performance_metrics[sender]["failures"] += 1
            self.agent_reliability[sender] *= 0.9
    
    def should_intervene(self) -> bool:
        """Determine if strategic intervention is needed."""
        # Check recent intervention history
        recent_window = 300  # 5 minutes
        recent_issues = [
            h for h in self.intervention_history
            if (datetime.now() - h["timestamp"]).seconds < recent_window
        ]
        
        # Intervention conditions
        if len(recent_issues) >= self.intervention_threshold:
            return True
        
        # Check for critical patterns
        if self._detect_critical_patterns(recent_issues):
            return True
        
        # Check system health
        if self._assess_system_health() == "critical":
            return True
        
        return False
    
    def _detect_critical_patterns(self, issues: List[Dict[str, Any]]) -> bool:
        """Detect critical patterns requiring intervention."""
        # Pattern 1: Same agent failing repeatedly
        agent_failures = defaultdict(int)
        for issue in issues:
            if issue.get("type") == "failure":
                agent_failures[issue["sender"]] += 1
        
        if any(count >= 3 for count in agent_failures.values()):
            return True
        
        # Pattern 2: Cascading conflicts
        conflict_chain = 0
        for i in range(1, len(issues)):
            if (issues[i]["type"] == "conflict" and 
                issues[i-1]["type"] == "conflict" and
                (issues[i]["timestamp"] - issues[i-1]["timestamp"]).seconds < 30):
                conflict_chain += 1
        
        if conflict_chain >= 2:
            return True
        
        # Pattern 3: System-wide degradation
        total_agents = len(self.agent_reliability)
        degraded_agents = sum(1 for r in self.agent_reliability.values() if r < 0.7)
        
        if total_agents > 0 and degraded_agents / total_agents > 0.5:
            return True
        
        return False
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        # Calculate health metrics
        avg_reliability = np.mean(list(self.agent_reliability.values())) if self.agent_reliability else 1.0
        
        total_successes = sum(m["successes"] for m in self.performance_metrics.values())
        total_failures = sum(m["failures"] for m in self.performance_metrics.values())
        
        if total_successes + total_failures > 0:
            success_rate = total_successes / (total_successes + total_failures)
        else:
            success_rate = 1.0
        
        # Determine health status
        if avg_reliability < 0.5 or success_rate < 0.3:
            return "critical"
        elif avg_reliability < 0.7 or success_rate < 0.6:
            return "degraded"
        else:
            return "healthy"
    
    async def strategic_intervention(self):
        """Execute strategic intervention."""
        try:
            # Analyze the situation
            analysis = await self._analyze_intervention_need()
            
            intervention = analysis.get("intervention_type", "none")
            
            if intervention == "challenge":
                await self._execute_challenge(analysis)
            elif intervention == "override":
                await self._execute_override(analysis)
            elif intervention == "escalate":
                await self._execute_escalation(analysis)
            else:
                logger.info("Controller analysis complete, no intervention needed")
            
            # Record intervention
            self.intervention_history.append({
                "timestamp": datetime.now(),
                "sender": self.agent_id,
                "content": analysis,
                "type": "intervention"
            })
            
        except Exception as e:
            logger.error(f"Controller intervention failed: {e}")
    
    async def _analyze_intervention_need(self) -> Dict[str, Any]:
        """Analyze the need for intervention."""
        # Prepare context
        context = {
            "recent_issues": self.intervention_history[-10:],
            "agent_reliability": dict(self.agent_reliability),
            "performance_metrics": dict(self.performance_metrics),
            "system_health": self._assess_system_health()
        }
        
        prompt = f"""Analyze system state and recommend intervention:

System Context:
{json.dumps(context, indent=2, default=str)}

Consider:
1. Pattern of failures or conflicts
2. Agent reliability trends
3. System-wide health
4. Whether intervention would help
5. What type of intervention is appropriate

Recommend the minimal intervention needed."""
        
        response = await self.call_llm(prompt)
        return json.loads(response)
    
    async def _execute_challenge(self, analysis: Dict[str, Any]):
        """Execute a challenge intervention."""
        target = analysis.get("target_of_intervention")
        
        await self.message_bus.publish(Message(
            type=MessageType.CHALLENGE,
            sender=self.agent_id,
            content={
                "intervention": "challenge",
                "target": target,
                "rationale": analysis["rationale"],
                "challenge_type": "verify_assumptions",
                "required_action": "Re-evaluate recent assertions with higher scrutiny"
            }
        ))
        
        logger.info(f"Controller challenged {target}: {analysis['rationale']}")
    
    async def _execute_override(self, analysis: Dict[str, Any]):
        """Execute an override intervention."""
        target = analysis.get("target_of_intervention")
        
        # Force specific agents to reset or change behavior
        await self.message_bus.publish(Message(
            type=MessageType.CHALLENGE,
            sender=self.agent_id,
            content={
                "intervention": "override",
                "target": target,
                "rationale": analysis["rationale"],
                "override_action": "reset_to_default",
                "new_parameters": {
                    "temperature": 0.3,  # More conservative
                    "confidence_threshold": 0.8  # Higher bar
                }
            }
        ))
        
        # Reset reliability score for fresh start
        if target in self.agent_reliability:
            self.agent_reliability[target] = 0.8
        
        logger.warning(f"Controller override on {target}: {analysis['rationale']}")
    
    async def _execute_escalation(self, analysis: Dict[str, Any]):
        """Execute an escalation intervention."""
        await self.message_bus.publish(Message(
            type=MessageType.CHALLENGE,
            sender=self.agent_id,
            content={
                "intervention": "escalate",
                "severity": "high",
                "rationale": analysis["rationale"],
                "system_health": analysis.get("system_health", {}),
                "recommendation": "Consider system restart or manual intervention"
            }
        ))
        
        logger.critical(f"Controller escalation: {analysis['rationale']}")

# ============== Interaction Layer Components ==============

class PersistentLog:
    """
    Persistent storage for knowledge and interactions.
    
    This component maintains a complete audit trail of all system activities,
    enabling replay, debugging, and compliance.
    """
    
    def __init__(self, storage_backend="postgresql"):
        self.storage_backend = storage_backend
        self.log_entries = []
        self.max_memory_entries = 10000
        self._init_storage()
        
    def _init_storage(self):
        """Initialize storage backend."""
        if self.storage_backend == "postgresql":
            # In production, initialize PostgreSQL connection
            pass
        elif self.storage_backend == "file":
            # Simple file-based storage for development
            self.log_file = "logs/persistent_log.jsonl"
            os.makedirs("logs", exist_ok=True)
    
    def log_interaction(
        self, 
        timestamp: datetime, 
        agent_id: str, 
        action: str, 
        data: Dict[str, Any]
    ):
        """Log an interaction with full context."""
        entry = {
            "timestamp": timestamp,
            "agent_id": agent_id,
            "action": action,
            "data": data,
            "id": hashlib.md5(
                f"{timestamp}_{agent_id}_{action}".encode()
            ).hexdigest()
        }
        
        self.log_entries.append(entry)
        
        # Persist to storage
        self._persist_entry(entry)
        
        # Manage memory usage
        if len(self.log_entries) > self.max_memory_entries:
            self.log_entries = self.log_entries[-self.max_memory_entries:]
    
    def _persist_entry(self, entry: Dict[str, Any]):
        """Persist entry to storage backend."""
        if self.storage_backend == "file":
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception as e:
                logger.error(f"Failed to persist log entry: {e}")
    
    def query_log(
        self, 
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query log entries with filters."""
        results = self.log_entries
        
        # Apply filters
        if agent_id:
            results = [e for e in results if e["agent_id"] == agent_id]
        if action:
            results = [e for e in results if e["action"] == action]
        if start_time:
            results = [e for e in results if e["timestamp"] >= start_time]
        if end_time:
            results = [e for e in results if e["timestamp"] <= end_time]
        
        # Sort by timestamp descending
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Apply limit
        return results[:limit]
    
    def get_agent_history(self, agent_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get complete history for a specific agent."""
        return self.query_log(agent_id=agent_id, limit=limit)
    
    def replay_from_checkpoint(self, checkpoint_time: datetime) -> List[Dict[str, Any]]:
        """Get all entries after a specific checkpoint for replay."""
        return self.query_log(start_time=checkpoint_time)

class TemporalTracker:
    """
    Track temporal logic and constraints.
    
    This component manages temporal relationships between events,
    enabling complex temporal reasoning and constraint checking.
    """
    
    def __init__(self):
        self.temporal_constraints = []
        self.event_sequence = []
        self.max_events = 1000
        
    def add_constraint(
        self, 
        constraint_type: str, 
        proposition: str, 
        scope: str = "always",
        duration: Optional[int] = None
    ):
        """
        Add temporal constraint.
        
        Args:
            constraint_type: Type of constraint (e.g., "consistency", "ordering")
            proposition: The proposition that must hold
            scope: Temporal scope ("always", "eventually", "until")
            duration: Duration in seconds (for time-bounded constraints)
        """
        self.temporal_constraints.append({
            "type": constraint_type,
            "proposition": proposition,
            "scope": scope,
            "duration": duration,
            "created_at": datetime.now(),
            "active": True
        })
        
        logger.debug(f"Added temporal constraint: {constraint_type} - {proposition}")
    
    def record_event(self, event: Dict[str, Any]):
        """Record temporal event."""
        event_record = {
            **event,
            "timestamp": datetime.now(),
            "id": hashlib.md5(
                f"{event.get('type', 'unknown')}_{datetime.now()}".encode()
            ).hexdigest()
        }
        
        self.event_sequence.append(event_record)
        
        # Maintain bounded sequence
        if len(self.event_sequence) > self.max_events:
            self.event_sequence = self.event_sequence[-self.max_events:]
        
        # Check constraints after each event
        violations = self.check_constraints()
        if violations:
            logger.warning(f"Temporal constraint violations: {len(violations)}")
    
    def check_constraints(self) -> List[Dict[str, Any]]:
        """Check if temporal constraints are satisfied."""
        violations = []
        current_time = datetime.now()
        
        for constraint in self.temporal_constraints:
            if not constraint["active"]:
                continue
            
            # Check time-bounded constraints
            if constraint["duration"]:
                age = (current_time - constraint["created_at"]).seconds
                if age > constraint["duration"]:
                    constraint["active"] = False
                    continue
            
            # Check constraint based on scope
            if constraint["scope"] == "always":
                if not self._check_always(constraint["proposition"]):
                    violations.append({
                        "constraint": constraint,
                        "violation_type": "always_violated",
                        "events": self._find_violating_events(constraint["proposition"])
                    })
                    
            elif constraint["scope"] == "eventually":
                if not self._check_eventually(constraint["proposition"], constraint["duration"]):
                    violations.append({
                        "constraint": constraint,
                        "violation_type": "not_yet_satisfied",
                        "time_remaining": constraint["duration"] - (current_time - constraint["created_at"]).seconds if constraint["duration"] else None
                    })
                    
            elif constraint["scope"] == "until":
                if not self._check_until(constraint["proposition"], constraint.get("until_condition")):
                    violations.append({
                        "constraint": constraint,
                        "violation_type": "until_violated"
                    })
        
        return violations
    
    def _check_always(self, proposition: str) -> bool:
        """Check if proposition holds in all events."""
        for event in self.event_sequence:
            if not self._evaluate_proposition(proposition, event):
                return False
        return True
    
    def _check_eventually(self, proposition: str, duration: Optional[int]) -> bool:
        """Check if proposition holds in at least one event."""
        for event in self.event_sequence:
            if self._evaluate_proposition(proposition, event):
                return True
        return False
    
    def _check_until(self, proposition: str, until_condition: Optional[str]) -> bool:
        """Check if proposition holds until another condition."""
        for event in self.event_sequence:
            if until_condition and self._evaluate_proposition(until_condition, event):
                return True
            if not self._evaluate_proposition(proposition, event):
                return False
        return True
    
    def _evaluate_proposition(self, proposition: str, event: Dict[str, Any]) -> bool:
        """
        Evaluate if proposition holds for event.
        
        This is a simplified implementation. In production, use a proper
        temporal logic evaluator.
        """
        # Simple keyword matching
        event_str = json.dumps(event).lower()
        prop_lower = proposition.lower()
        
        # Check for negation
        if prop_lower.startswith("not "):
            return prop_lower[4:] not in event_str
        
        # Check for conjunction
        if " and " in prop_lower:
            parts = prop_lower.split(" and ")
            return all(part.strip() in event_str for part in parts)
        
        # Check for disjunction
        if " or " in prop_lower:
            parts = prop_lower.split(" or ")
            return any(part.strip() in event_str for part in parts)
        
        # Simple containment check
        return prop_lower in event_str
    
    def _find_violating_events(self, proposition: str) -> List[Dict[str, Any]]:
        """Find events that violate the proposition."""
        violating = []
        
        for event in self.event_sequence[-10:]:  # Last 10 events
            if not self._evaluate_proposition(proposition, event):
                violating.append({
                    "event_id": event.get("id"),
                    "timestamp": event.get("timestamp"),
                    "type": event.get("type", "unknown")
                })
        
        return violating
    
    def get_event_timeline(self, time_window: int = 300) -> List[Dict[str, Any]]:
        """Get timeline of recent events."""
        current_time = datetime.now()
        recent_events = [
            e for e in self.event_sequence
            if (current_time - e["timestamp"]).seconds <= time_window
        ]
        
        return sorted(recent_events, key=lambda x: x["timestamp"])

# ============== Main ReAgent System ==============

class ReAgentSystem:
    """
    Main orchestrator for the ReAgent multi-agent system.
    
    This class coordinates all agents, manages the message bus,
    and provides the main interface for processing questions.
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.llm_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Initialize message bus
        self.message_bus = MessageBus()
        
        # Initialize interaction layer components
        self.persistent_log = PersistentLog()
        self.temporal_tracker = TemporalTracker()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Update metrics
        metrics_collector.update_active_agents(len(self.agents))
        
        # Set up temporal constraints
        self._setup_temporal_constraints()
        
        # System state
        self.is_running = False
        self.message_processor_task = None
        
        logger.info(f"ReAgent system initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all agents."""
        agents = {
            'decomposer': QuestionDecomposerAgent('A_Q', self.llm_client, self.message_bus),
            'retriever': RetrieverAgent('A_R', self.llm_client, self.message_bus),
            'verifier': VerifierAgent('A_V', self.llm_client, self.message_bus),
            'assembler': AnswerAssemblerAgent('A_A', self.llm_client, self.message_bus),
            'supervisor': SupervisorAgent('A_S', self.llm_client, self.message_bus),
            'controller': ControllerAgent('A_C', self.llm_client, self.message_bus)
        }
        
        return agents
    
    def _setup_temporal_constraints(self):
        """Set up system-wide temporal constraints."""
        # No unresolved conflicts should persist
        self.temporal_tracker.add_constraint(
            "consistency",
            "no_unresolved_conflicts",
            "always"
        )
        
        # Questions should be answered within time limit
        self.temporal_tracker.add_constraint(
            "performance",
            "question_answered",
            "eventually",
            duration=300  # 5 minutes
        )
        
        # System should maintain minimum reliability
        self.temporal_tracker.add_constraint(
            "reliability",
            "system_healthy",
            "always"
        )
    
    async def start(self):
        """Start the ReAgent system."""
        if self.is_running:
            logger.warning("System already running")
            return
        
        self.is_running = True
        
        # Start message processing
        self.message_processor_task = asyncio.create_task(
            self.message_bus.process_messages()
        )
        
        logger.info("ReAgent system started")
    
    async def stop(self):
        """Stop the ReAgent system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel message processing
        if self.message_processor_task:
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("ReAgent system stopped")
    
    async def process_question(self, question: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Process a multi-hop question through the system.
        
        Args:
            question: The question to process
            timeout: Maximum time to wait for answer (seconds)
            
        Returns:
            Dictionary containing the answer and reasoning trace
        """
        start_time = datetime.now()
        question_id = hashlib.md5(
            f"{question}_{start_time}".encode()
        ).hexdigest()
        
        logger.info(f"Processing question {question_id[:8]}: {question[:50]}...")
        
        # Log question start
        self.persistent_log.log_interaction(
            start_time,
            "system",
            "question_received",
            {"question": question, "question_id": question_id}
        )
        
        # Record event
        self.temporal_tracker.record_event({
            "type": "question_start",
            "question_id": question_id
        })
        
        # Ensure system is running
        if not self.is_running:
            await self.start()
        
        try:
            # Send initial question to decomposer
            await self.message_bus.publish(Message(
                type=MessageType.ASSERT,
                sender="user",
                content={
                    "original_question": question,
                    "question_id": question_id
                }
            ))
            
            # Wait for processing with timeout
            result = await self._wait_for_answer(question_id, timeout)
            
            # Record metrics
            duration = (datetime.now() - start_time).total_seconds()
            success = result.get("status") == "success"
            metrics_collector.record_question_processed(success, duration)
            
            # Log completion
            self.persistent_log.log_interaction(
                datetime.now(),
                "system",
                "question_completed",
                {
                    "question_id": question_id,
                    "duration": duration,
                    "success": success
                }
            )
            
            # Record completion event
            self.temporal_tracker.record_event({
                "type": "question_complete",
                "question_id": question_id,
                "success": success
            })
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Question processing timed out after {timeout}s")
            
            # Record failure
            duration = (datetime.now() - start_time).total_seconds()
            metrics_collector.record_question_processed(False, duration)
            
            return {
                "status": "timeout",
                "error": f"Processing timed out after {timeout} seconds",
                "partial_results": self._collect_partial_results()
            }
            
        except Exception as e:
            logger.error(f"System error: {e}")
            
            # Record failure
            duration = (datetime.now() - start_time).total_seconds()
            metrics_collector.record_question_processed(False, duration)
            
            return {
                "status": "error",
                "error": str(e),
                "trace": self._build_error_trace(start_time)
            }
    
    async def _wait_for_answer(self, question_id: str, timeout: int) -> Dict[str, Any]:
        """Wait for final answer with timeout."""
        end_time = time.time() + timeout
        check_interval = 0.5  # Check every 500ms
        
        while time.time() < end_time:
            # Check for final answer in logs
            answer_entries = self.persistent_log.query_log(
                agent_id="A_A",
                action="final_answer",
                limit=10
            )
            
            for entry in answer_entries:
                if entry["data"].get("question_id") == question_id:
                    return {
                        "status": "success",
                        "answer": entry["data"].get("final_answer"),
                        "confidence": entry["data"].get("confidence", 0.8),
                        "reasoning_trace": self._build_reasoning_trace(question_id),
                        "supporting_facts": entry["data"].get("supporting_facts", []),
                        "metadata": entry["data"].get("metadata", {})
                    }
            
            # Check for failures
            failure_entries = self.persistent_log.query_log(
                action="failure",
                limit=10
            )
            
            for entry in failure_entries:
                if entry["data"].get("question_id") == question_id:
                    return {
                        "status": "failed",
                        "error": entry["data"].get("error", "Unknown error"),
                        "partial_results": self._collect_partial_results()
                    }
            
            await asyncio.sleep(check_interval)
        
        # Timeout reached
        raise asyncio.TimeoutError()
    
    def _build_reasoning_trace(self, question_id: str = None) -> List[Dict[str, Any]]:
        """Build complete reasoning trace from logs."""
        trace = []
        
        # Get all relevant log entries
        entries = self.persistent_log.query_log(limit=1000)
        
        # Filter by question_id if provided
        if question_id:
            entries = [
                e for e in entries
                if e.get("data", {}).get("question_id") == question_id
            ]
        
        # Build trace
        for entry in sorted(entries, key=lambda x: x["timestamp"]):
            trace_item = {
                "timestamp": entry["timestamp"].isoformat(),
                "agent": entry["agent_id"],
                "action": entry["action"],
                "summary": self._summarize_entry(entry)
            }
            
            # Add key data points
            if "sub_questions" in entry.get("data", {}):
                trace_item["sub_questions"] = entry["data"]["sub_questions"]
            if "verified_facts" in entry.get("data", {}):
                trace_item["facts"] = entry["data"]["verified_facts"]
            if "conflicts" in entry.get("data", {}):
                trace_item["conflicts"] = entry["data"]["conflicts"]
            
            trace.append(trace_item)
        
        return trace
    
    def _summarize_entry(self, entry: Dict[str, Any]) -> str:
        """Create summary of log entry."""
        action = entry.get("action", "unknown")
        agent = entry.get("agent_id", "unknown")
        
        if action == "decompose_question":
            return f"{agent} decomposed question into sub-questions"
        elif action == "retrieve_evidence":
            return f"{agent} retrieved evidence"
        elif action == "verify_evidence":
            return f"{agent} verified evidence"
        elif action == "local_backtrack":
            return f"{agent} performed local backtracking"
        elif action == "global_rollback":
            return f"{agent} initiated global rollback"
        elif action == "final_answer":
            return f"{agent} assembled final answer"
        else:
            return f"{agent} performed {action}"
    
    def _collect_partial_results(self) -> Dict[str, Any]:
        """Collect partial results from all agents."""
        results = {}
        
        for agent_id, agent in self.agents.items():
            if agent.local_knowledge.assertions:
                results[agent_id] = {
                    "assertions": [
                        {
                            "content": a.content,
                            "confidence": a.confidence,
                            "source": a.source
                        }
                        for a in agent.local_knowledge.assertions.values()
                    ],
                    "checkpoint_count": len(agent.local_knowledge.checkpoints),
                    "performance": agent.get_performance_stats()
                }
        
        return results
    
    def _build_error_trace(self, start_time: datetime) -> List[Dict[str, Any]]:
        """Build error trace for debugging."""
        # Get all entries since start
        entries = self.persistent_log.query_log(
            start_time=start_time,
            limit=1000
        )
        
        # Focus on errors and conflicts
        error_trace = []
        
        for entry in entries:
            if entry["action"] in ["error", "failure", "conflict", "challenge"]:
                error_trace.append({
                    "timestamp": entry["timestamp"].isoformat(),
                    "agent": entry["agent_id"],
                    "type": entry["action"],
                    "details": entry.get("data", {})
                })
        
        return error_trace
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        agent_status = {}
        
        for agent_id, agent in self.agents.items():
            agent_status[agent_id] = {
                "assertions": len(agent.local_knowledge.assertions),
                "checkpoints": len(agent.local_knowledge.checkpoints),
                "performance": agent.get_performance_stats()
            }
        
        return {
            "is_running": self.is_running,
            "agents": agent_status,
            "temporal_constraints": len(self.temporal_tracker.temporal_constraints),
            "constraint_violations": len(self.temporal_tracker.check_constraints()),
            "recent_events": len(self.temporal_tracker.event_sequence),
            "log_entries": len(self.persistent_log.log_entries)
        }

# ============== CLI Interface ==============

async def main():
    """Example usage of ReAgent system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system
    system = ReAgentSystem()
    
    # Start system
    await system.start()
    
    try:
        # Example questions
        questions = [
            "Which U.S. state has a capital city whose population is smaller than the state's largest city, given that this state hosted the 1984 Summer Olympics?",
            "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?",
            "Who was the president of the United States when the company that created the iPhone was founded?"
        ]
        
        # Process questions
        for question in questions[:1]:  # Process first question as example
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"{'='*80}\n")
            
            result = await system.process_question(question, timeout=60)
            
            print(f"\nResult Status: {result['status']}")
            
            if result['status'] == 'success':
                print(f"Answer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"\nSupporting Facts:")
                for fact in result.get('supporting_facts', []):
                    print(f"  - {fact}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print(f"\n{'='*80}")
        
        # Show system status
        status = system.get_system_status()
        print(f"\nSystem Status:")
        print(json.dumps(status, indent=2))
        
    finally:
        # Stop system
        await system.stop()

if __name__ == "__main__":
    asyncio.run(main())