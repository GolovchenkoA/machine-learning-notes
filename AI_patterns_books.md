# AI Patterns: Books and Courses

Caveat: the search did not return the LinkedIn Learning catalog, so verify the course titles there. The books are well established.

## The patterns that make AI software reliable, fast and cheap

There is no Gang of Four book for LLM systems yet. These are the recurring patterns:

- **Reliability:** evals as unit tests, structured outputs with schema validation, guardrails, retries and fallbacks between models, human escalation paths.
- **Speed:** streaming, response and prompt caching, parallel calls, smaller specialist models.
- **Cost:** model routing and cascades (cheap model first, escalate only when needed), semantic caching, batching, context trimming.
- **Capability:** RAG, tool use and function calling, ReAct-style agents, reflection and self-critique loops.

## 4 books

1. **AI Engineering** by Chip Huyen (O'Reilly, 2025). Start here. Reviewers rank it first for developers entering AI engineering in 2026, and it covers evaluation, inference optimization and deployment.
2. **Machine Learning Design Patterns** by Lakshmanan, Robinson and Munn (O'Reilly, 2020). It is the only true pattern catalog of the four, and it's listed as a reusable pattern catalog among the top MLOps books. It predates LLMs, but patterns like Cascade, Fallback and Reframing still apply.
3. **LLM Engineer's Handbook** by Paul Iusztin and Maxime Labonne. It builds an end-to-end LLM system. One list suggests starting with it if you're short on time.
4. **Designing Machine Learning Systems** by Chip Huyen (O'Reilly, 2022). It is the foundation for data drift, monitoring and retraining. It's described as the best single explanation of why ML systems fail in production.

Optional fifth: *LLMOps* (O'Reilly, 2025), which focuses on running LLMs in production.

## 4 LinkedIn Learning courses

Verify these on LinkedIn Learning, since titles and availability change.

1. **Generative AI: Working with Large Language Models** (Jonathan Fernandes), if it's still listed. It covers how LLMs behave and how to use them via APIs, and it's the foundation for the rest.
2. **Hands-On AI: Building LLM-Powered Apps** (Harpreet Sahota), if it's still listed. It covers RAG and LangChain-style app building. Search "RAG", "LangChain" or "vector database" if the title is gone.
3. **A prompt engineering course for developers.** Search "Prompt Engineering" and pick one that covers structured output, few-shot examples and prompt evaluation, not just chatting tips.
4. **An LLMOps or AI agents course.** Search "LLMOps", "AI agents" or "evaluating LLM applications". This is the thinnest area on LinkedIn Learning, so look for anything on evals, monitoring and guardrails.

LinkedIn Learning is stronger on fundamentals than on production patterns. For evals, caching and routing, the books above and a hands-on course such as Frontend Masters' *Production AI* (evals, RAG and structured outputs) will go deeper.

## Suggested order

Read *AI Engineering* while doing the LLM and RAG courses, then *ML Design Patterns* for the pattern vocabulary. Build one small project with evals, a cheap-to-expensive model cascade and a cache, and use *Designing ML Systems* as a reference for monitoring.

## Sources

- [I Have Read 20+ Books on AI and LLM Engineering](https://reactjava.substack.com/p/i-have-read-20-books-on-ai-and-llm)
- [8 Must-Read AI Books for Developers and Engineers in 2026](https://travis.media/blog/8-must-read-ai-tech-books/)
- [Best MLOps Books](https://computingforgeeks.com/best-mlops-books/)
- [Production AI, Frontend Masters](https://frontendmasters.com/courses/production-ai/)
