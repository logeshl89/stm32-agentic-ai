"""
Agentic RAG System for STM32F446RE Documentation
Implements advanced reasoning and query processing capabilities
"""
from typing import List, Dict, Any, Optional
from src.retrieval.retriever import Retriever, RetrievalStrategy
from src.agentic_layer.answer_generator import AnswerGenerator
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """
    Possible states of the agentic system
    """
    PROCESSING = "processing"
    RETRIEVING = "retrieving"
    REASONING = "reasoning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryType(Enum):
    """
    Types of queries that can be processed
    """
    SIMPLE_FACT = "simple_fact"
    COMPLEX_REASONING = "complex_reasoning"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    TROUBLESHOOTING = "troubleshooting"


class AgenticRAG:
    """
    Main agentic system that coordinates retrieval, reasoning, and generation
    """
    
    def __init__(self, retriever: Retriever, llm_interface=None):
        """
        Initialize the agentic system
        
        Args:
            retriever: The retrieval system
            llm_interface: Interface to the language model (uses mock if None)
        """
        self.retriever = retriever
        self.answer_generator = llm_interface or AnswerGenerator()
        self.query_analyzer = QueryAnalyzer()
        self.state = AgentState.PROCESSING
        self.conversation_history = []
        
    def process_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query through the agentic system
        
        Args:
            query (str): User query
            user_context (Dict[str, Any]): Additional context about the user
            
        Returns:
            Dict[str, Any]: Processed response with citations and confidence
        """
        logger.info(f"Processing query: {query}")
        
        self.state = AgentState.PROCESSING
        response = {
            'query': query,
            'query_type': None,
            'retrieved_docs': [],
            'answer': '',
            'confidence': 0.0,
            'citations': [],
            'reasoning_trace': [],
            'followup_questions': [],
            'abstained': False,
            'abstain_reason': '',
            'next_best_queries': [],
            'retrieval_strategy': None,
            'retrieval_filters': {},
            'state': self.state.value
        }
        
        try:
            # Analyze the query
            self.state = AgentState.REASONING
            query_analysis = self.query_analyzer.analyze(query)
            response['query_type'] = query_analysis['type']
            response['reasoning_trace'].append(f"Query classified as: {query_analysis['type']}")
            
            # Retrieve relevant documents
            self.state = AgentState.RETRIEVING
            k = self._determine_retrieval_k(query_analysis)
            strategy = self._select_retrieval_strategy(query_analysis)
            min_similarity = self._determine_min_similarity(query_analysis)
            filters = self._build_filters(query_analysis)

            retrieved_docs = self.retriever.retrieve(
                query,
                k=k,
                strategy=strategy,
                filters=filters,
                rerank=True,
                min_similarity=min_similarity
            )

            response['retrieved_docs'] = retrieved_docs
            response['retrieval_strategy'] = strategy.value
            response['retrieval_filters'] = filters
            response['reasoning_trace'].append(f"Retrieved {len(retrieved_docs)} relevant documents with {strategy.value}")
            
            # Generate answer based on retrieved context
            self.state = AgentState.GENERATING
            answer_result = self._generate_answer(query, retrieved_docs, query_analysis)
            
            response['answer'] = answer_result['answer']
            response['confidence'] = answer_result['confidence']
            response['citations'] = answer_result['citations']
            response['followup_questions'] = answer_result['followup_questions']
            response['abstained'] = answer_result.get('abstained', False)
            response['abstain_reason'] = answer_result.get('abstain_reason', '')
            response['next_best_queries'] = answer_result.get('next_best_queries', [])
            
            # Log the interaction
            self._log_interaction(query, response)
            
            self.state = AgentState.COMPLETED
            response['state'] = self.state.value
            
            logger.info(f"Query processed successfully. Confidence: {response['confidence']}")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            self.state = AgentState.FAILED
            response['state'] = self.state.value
            response['error'] = str(e)
            response['answer'] = "An error occurred while processing your query. Please try again."
        
        return response
    
    def _determine_retrieval_k(self, query_analysis: Dict[str, Any]) -> int:
        """Use query complexity/type to pick dynamic top-k."""
        query_type = query_analysis.get('type')
        complexity = query_analysis.get('complexity', 'medium')

        if query_type == QueryType.SIMPLE_FACT:
            return 4 if complexity == 'low' else 5
        if query_type == QueryType.PROCEDURAL:
            return 8 if complexity == 'high' else 6
        if query_type in (QueryType.COMPARATIVE, QueryType.TROUBLESHOOTING):
            return 8
        return 7

    def _determine_min_similarity(self, query_analysis: Dict[str, Any]) -> float:
        """Tune retrieval threshold by query type."""
        query_type = query_analysis.get('type')
        if query_type == QueryType.SIMPLE_FACT:
            return 0.35
        if query_type == QueryType.TROUBLESHOOTING:
            return 0.2
        return 0.25

    def _select_retrieval_strategy(self, query_analysis: Dict[str, Any]) -> RetrievalStrategy:
        """Select retrieval strategy based on query intent."""
        query_type = query_analysis.get('type')
        if query_type in (QueryType.PROCEDURAL, QueryType.COMPARATIVE, QueryType.TROUBLESHOOTING):
            return RetrievalStrategy.HYBRID
        return RetrievalStrategy.SEMANTIC

    def _build_filters(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build metadata filters based on query analysis
        
        Args:
            query_analysis: Result of query analysis
            
        Returns:
            Dict[str, Any]: Filters to apply during retrieval
        """
        filters = {}
        
        # Add filters based on query type
        if query_analysis['type'] == QueryType.TROUBLESHOOTING:
            filters['section_type'] = 'general_content'  # Troubleshooting often spans sections
        elif query_analysis['type'] == QueryType.COMPARATIVE:
            filters['section_type'] = 'general_content'
        elif query_analysis['type'] == QueryType.PROCEDURAL:
            filters['section_type'] = 'peripheral_description'
        
        # Add specific hardware component filters if identified
        components = query_analysis.get('components', [])
        peripherals = [c.upper() for c in components if c.upper() in {'GPIO', 'USART', 'SPI', 'I2C', 'TIM', 'ADC', 'DAC', 'DMA'}]
        pins = [c.upper() for c in components if c.upper().startswith('P') and len(c) >= 3]

        if peripherals:
            filters['peripheral'] = peripherals[0]
        if pins:
            filters['pin'] = pins[0]
        
        return filters
    
    def _generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                        query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an answer based on the query and retrieved documents
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            query_analysis: Analysis of the query
            
        Returns:
            Dict[str, Any]: Generated answer with metadata
        """
        if not retrieved_docs:
            # Handle basic questions that don't require documentation
            return self._handle_basic_questions(query)
        
        # Generate answer using the answer generator
        answer_result = self.answer_generator.generate_answer(query, retrieved_docs, query_analysis['type'].value if hasattr(query_analysis['type'], 'value') else str(query_analysis['type']))
        
        # Extract answer, confidence, and citations
        answer = answer_result.get('answer', '')
        confidence = answer_result.get('confidence', 0.5)
        # Format citations from retrieved docs
        citations = []
        for doc in retrieved_docs[:3]:  # Top 3 citations
            citation = {
                'source': doc['metadata'].get('section_header', 'N/A'),
                'pages': doc['metadata'].get('pages', []),
                'section_type': doc['metadata'].get('section_type', 'N/A'),
                'similarity': doc.get('similarity', 0.0)
            }
            citations.append(citation)
        
        # Generate follow-up questions
        followup_questions = self._generate_followup_questions(query, answer)
        
        abstained = confidence < 0.3
        return {
            'answer': answer,
            'confidence': confidence,
            'citations': citations,
            'followup_questions': followup_questions,
            'abstained': abstained,
            'abstain_reason': 'low_confidence' if abstained else '',
            'next_best_queries': self._fallback_next_best_queries(query, query_analysis)
        }
    
    def _fallback_next_best_queries(self, query: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Suggest best-next refinements when confidence is low."""
        hints = []
        components = query_analysis.get('components', [])
        if components:
            hints.append(f"Can you share if you mean {'/'.join(components[:2])} specifically?")
        hints.extend([
            "Can you include exact register names or bit fields?",
            "Do you want initialization steps or troubleshooting guidance?",
            "Can you provide the peripheral and MCU pin mapping details?"
        ])
        return hints[:3]

    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from retrieved documents
        
        Args:
            retrieved_docs: Retrieved documents
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            chunk_content = doc['chunk']
            metadata = doc['metadata']
            
            context_part = f"[Document {i}] "
            context_part += f"Section: {metadata.get('section_header', 'N/A')}, "
            context_part += f"Pages: {', '.join(map(str, metadata.get('pages', [])))}, "
            context_part += f"Type: {metadata.get('section_type', 'N/A')}\n"
            context_part += f"Content: {chunk_content}\n\n"
            
            context_parts.append(context_part)
        
        return "".join(context_parts)
    
    def _handle_basic_questions(self, query: str) -> Dict[str, Any]:
        """
        Handle basic questions that don't require documentation lookup
        
        Args:
            query: User query
            
        Returns:
            Dict[str, Any]: Generated response
        """
        query_lower = query.lower()
        
        # Handle greetings
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return {
                'answer': "Hello! I'm your STM32F446RE documentation assistant. I can help you with questions about STM32F446RE microcontroller registers, peripherals, configuration, and technical documentation.",
                'confidence': 0.95,
                'citations': [],
                'followup_questions': ["What would you like to know about the STM32F446RE?", "Do you need help with GPIO configuration?", "Are you working on a specific peripheral?"],
                'abstained': False,
                'abstain_reason': '',
                'next_best_queries': []
            }
        
        # Handle capability questions
        elif any(phrase in query_lower for phrase in ['what can you', 'what do you know', 'help with', 'capable of']):
            return {
                'answer': "I can help you with STM32F446RE microcontroller documentation, including: GPIO configuration, peripheral registers, clock settings, interrupt handling, power management, memory organization, and specific technical procedures. I can answer questions about register addresses, bit configurations, and implementation examples based on the official STM32F446RE reference manual.",
                'confidence': 0.95,
                'citations': [],
                'followup_questions': ["What specific topic would you like to explore?", "Do you need help with a particular peripheral?", "Are you working on a configuration issue?"]
            }
        
        # Handle identity questions
        elif any(phrase in query_lower for phrase in ['who are you', 'what are you', 'introduce yourself']):
            return {
                'answer': "I'm an AI assistant specifically designed for STM32F446RE microcontroller documentation. I can answer technical questions about registers, peripherals, and configuration based on the official STM32F446RE reference documentation. I provide cited answers with page numbers and section references.",
                'confidence': 0.95,
                'citations': [],
                'followup_questions': ["What STM32F446RE topic would you like to discuss?", "Do you have a specific technical question?"]
            }
        
        # Handle general microcontroller questions
        elif any(phrase in query_lower for phrase in ['what is a microcontroller', 'what is stm32', 'arm cortex', 'cortex-m4']):
            return {
                'answer': "The STM32F446RE is an ARM Cortex-M4 based 32-bit microcontroller with a maximum frequency of 180 MHz. It features 512 KB of Flash memory, 128 KB of SRAM, multiple peripherals including GPIOs, USART, SPI, I2C, timers, ADC, DAC, and supports various power modes for efficient battery operation.",
                'confidence': 0.85,
                'citations': [],
                'followup_questions': ["Do you need specific register information?", "What peripheral would you like to configure?", "Do you have a programming question?"]
            }
        
        # Return generic low confidence response
        return {
            'answer': "I don't have specific information about that topic in the STM32F446RE documentation. Please try rephrasing your question or ask about specific registers, peripherals, or configuration topics related to the STM32F446RE.",
            'confidence': 0.1,
            'citations': [],
            'followup_questions': [],
            'abstained': True,
            'abstain_reason': 'insufficient_documentation_context',
            'next_best_queries': [
                "Which STM32F446RE peripheral are you using?",
                "Can you share the exact register name or pin?",
                "Should I focus on setup steps, troubleshooting, or comparison?"
            ]
        }

    def _construct_prompt(self, query: str, context: str, query_analysis: Dict[str, Any]) -> str:
        """
        Construct the prompt for the LLM
        
        Args:
            query: User query
            context: Retrieved context
            query_analysis: Analysis of the query
            
        Returns:
            str: Formatted prompt
        """
        prompt_template = """
You are an expert assistant for the STM32F446RE microcontroller. Answer the user's query based strictly on the provided documentation context. Do not hallucinate or make up information.

Documentation Context:
{context}

User Query: {query}

Query Type: {query_type}

Instructions:
1. Answer only based on the provided context
2. If the answer is not in the context, clearly state this
3. Provide specific citations to the documentation
4. For technical questions, be precise and accurate
5. If applicable, mention page numbers and section headers
6. Focus on STM32F446RE specific features and capabilities

Answer:
"""
        
        return prompt_template.format(
            context=context,
            query=query,
            query_type=query_analysis['type'].value
        )
    
    def _generate_followup_questions(self, query: str, answer: str) -> List[str]:
        """
        Generate potential follow-up questions
        
        Args:
            query: Original query
            answer: Generated answer
            
        Returns:
            List[str]: Potential follow-up questions
        """
        # This would use an LLM to generate follow-up questions in a real implementation
        # For now, we'll use a simple heuristic approach
        followups = []
        
        # Example follow-up generation based on query patterns
        if "configure" in query.lower() or "set up" in query.lower():
            followups.extend([
                "What are the possible configuration options?",
                "Are there any prerequisites for this configuration?",
                "How do I verify the configuration is correct?"
            ])
        elif "difference" in query.lower() or "compare" in query.lower():
            followups.extend([
                "What are the practical implications of these differences?",
                "Which option is recommended for my use case?"
            ])
        
        return followups[:3]  # Return top 3 follow-ups
    
    def _log_interaction(self, query: str, response: Dict[str, Any]):
        """
        Log the interaction for analytics and improvement
        
        Args:
            query: User query
            response: System response
        """
        interaction = {
            'query': query,
            'query_type': response['query_type'],
            'confidence': response['confidence'],
            'num_retrieved_docs': len(response['retrieved_docs']),
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        self.conversation_history.append(interaction)


class QueryAnalyzer:
    """
    Analyzes user queries to determine type and required processing
    """
    
    def __init__(self):
        self.simple_keywords = {
            'what', 'how', 'when', 'where', 'which', 'who', 'why',
            'define', 'explain', 'describe', 'tell me about'
        }

        self.procedural_keywords = {
            'configure', 'set up', 'setup', 'initialize', 'enable', 'disable',
            'connect', 'implement', 'use', 'program'
        }

        self.comparative_keywords = {
            'difference', 'compare', 'versus', 'vs', 'between',
            'similarities', 'better', 'alternative'
        }

        self.troubleshooting_keywords = {
            'problem', 'issue', 'debug', 'fix', 'solution',
            'doesn\'t work', 'not working', 'error', 'help'
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine its type and characteristics
        
        Args:
            query: User query
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Determine query type
        if self._matches_keywords(query_lower, words, self.troubleshooting_keywords):
            query_type = QueryType.TROUBLESHOOTING
        elif self._matches_keywords(query_lower, words, self.comparative_keywords):
            query_type = QueryType.COMPARATIVE
        elif self._matches_keywords(query_lower, words, self.procedural_keywords):
            query_type = QueryType.PROCEDURAL
        elif self._matches_keywords(query_lower, words[:3], self.simple_keywords):
            query_type = QueryType.SIMPLE_FACT
        else:
            query_type = QueryType.COMPLEX_REASONING
        
        # Extract potential components (registers, peripherals, etc.)
        components = self._extract_components(query)
        
        return {
            'type': query_type,
            'components': components,
            'length': len(words),
            'complexity': self._estimate_complexity(query)
        }

    def _matches_keywords(self, query_lower: str, tokens: List[str], keywords: set) -> bool:
        """Match both single-token and phrase keywords against a query."""
        for keyword in keywords:
            if " " in keyword:
                if keyword in query_lower:
                    return True
            elif keyword in tokens:
                return True
        return False
    
    def _extract_components(self, query: str) -> List[str]:
        """
        Extract potential hardware components from the query
        
        Args:
            query: User query
            
        Returns:
            List[str]: Extracted components
        """
        import re
        
        # Patterns for common STM32 components
        patterns = [
            r'\b(GPIO|USART|SPI|I2C|TIM|ADC|DAC|DMA)\b',  # Peripherals
            r'\b(P[A-Z][0-9]+)\b',  # Pin names like PA0, PB5
            r'\b([A-Z0-9_]+_R[A-Z0-9_]*)\b',  # Register names
            r'\b(0x[0-9A-Fa-f]{2,8})\b',  # Addresses
        ]
        
        components = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            components.extend(matches)
        
        return list(set(components))  # Remove duplicates
    
    def _estimate_complexity(self, query: str) -> str:
        """
        Estimate the complexity of the query
        
        Args:
            query: User query
            
        Returns:
            str: Complexity level
        """
        length = len(query.split())
        
        if length <= 5:
            return 'low'
        elif length <= 15:
            return 'medium'
        else:
            return 'high'


class MockLLMInterface:
    """
    Mock interface for the LLM - to be replaced with actual implementation
    """
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from the LLM (mock implementation)
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dict[str, Any]: Generated response
        """
        # In a real implementation, this would call an actual LLM API
        # For now, we return a mock response
        
        import random
        
        # Extract the actual query from the prompt
        lines = prompt.split('\n')
        query_line = ""
        for line in lines:
            if line.startswith('User Query:'):
                query_line = line.replace('User Query:', '').strip()
                break
        
        # Generate a mock response based on the query
        if "configure" in query_line.lower() or "setup" in query_line.lower():
            answer = f"Based on the STM32F407 documentation, to {query_line.lower()}, you would need to follow these steps: 1) Initialize the appropriate registers, 2) Configure the required settings, 3) Enable the peripheral. See the referenced documentation sections for specific register details."
        elif "what is" in query_line.lower() or "define" in query_line.lower():
            answer = f"The STM32F407 documentation describes this as a specific feature or component of the microcontroller. Please refer to the cited sections for detailed technical specifications."
        else:
            answer = f"According to the STM32F407 documentation, this topic is covered in the referenced sections. The specific implementation details depend on your particular use case and requirements."
        
        return {
            'answer': answer,
            'confidence': round(random.uniform(0.6, 0.9), 2),
            'raw_response': 'Mock response for demonstration purposes'
        }


class ToolInterface:
    """
    Interface for external tools that the agent can use
    """
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, name: str, tool_func):
        """
        Register a new tool
        
        Args:
            name: Tool name
            tool_func: Function implementing the tool
        """
        self.tools[name] = tool_func
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a registered tool
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments for the tool
            
        Returns:
            Any: Tool result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        return self.tools[tool_name](**kwargs)


# Example usage
if __name__ == "__main__":
    # Example usage would require initialized components
    # vector_store = VectorStoreManager()
    # embedder = Embedder()
    # retriever = Retriever(vector_store, embedder)
    # agent = AgenticRAG(retriever)
    #
    # print("Agentic RAG system initialized")
    pass
