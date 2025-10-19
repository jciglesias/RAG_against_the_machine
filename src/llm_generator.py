import subprocess
from typing import List, Optional
from pathlib import Path

from src.models import MinimalSource
from src.config import DEFAULT_MODEL


class LLMGenerator:

    def __init__(self, model: str = DEFAULT_MODEL, max_context_chars: int = 8000):
        self.model = model
        self.max_context_chars = max_context_chars

    def generate_answer(
            self,
            question: str,
            sources: List[MinimalSource],
            repo_path: Path) -> str:
        context = self._build_context(sources, repo_path)

        prompt = self._create_prompt(question, context)

        try:
            answer = self._call_llm(prompt)
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def _build_context(self, sources: List[MinimalSource], repo_path: Path) -> str:
        context_parts = []
        current_length = 0

        for i, source in enumerate(sources):
            try:
                file_path = repo_path / source.file_path

                if not file_path.exists():
                    continue

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                chunk = content[
                    source.first_character_index:source.last_character_index
                ]

                source_text = (
                    f"\n--- Source {i+1}: {source.file_path} ---\n{chunk}\n"
                )

                if current_length + len(source_text) > \
                        self.max_context_chars:
                    available_space = (
                        self.max_context_chars - current_length - 100
                    )
                    if available_space > 200:
                        chunk_truncated = (
                            chunk[:available_space] + "\n[... truncated ...]"
                        )
                        source_text = (
                            f"\n--- Source {i+1}: {source.file_path} ---\n"
                            f"{chunk_truncated}\n"
                        )
                        context_parts.append(source_text)
                    break

                context_parts.append(source_text)
                current_length += len(source_text)

            except Exception as e:
                print(f"Error reading source {source.file_path}: {e}")
                continue

        return ''.join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful AI assistant that answers questions \
about the vLLM codebase.

Based on the following code and documentation snippets, please answer the \
question accurately and concisely.

Question: {question}

Context:
{context}

Instructions:
- Answer based ONLY on the provided context
- Be specific and reference file names or code when relevant
- If the context doesn't contain enough information, say so
- Keep your answer clear and concise
- Use code examples from the context when appropriate

Answer:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        if self.model.startswith("ollama"):
            return self._call_ollama(prompt)
        else:
            return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        try:
            if "/" in self.model:
                model_name = self.model.split("/", 1)[1]
            else:
                model_name = "qwen3:0.6b"

            cmd = ["ollama", "run", model_name]

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = result.stderr or "Unknown error"
                return (
                    f"Error calling Ollama: {error_msg}"
                )

        except FileNotFoundError:
            return (
                "Error: Ollama not found. Please install ollama or "
                "use a different model."
            )
        except subprocess.TimeoutExpired:
            return "Error: Ollama request timed out (30s limit)"
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"

    def _call_openai_compatible(
            self,
            prompt: str,
            api_url: str,
            api_key: Optional[str] = None) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }

        try:
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error calling API: {str(e)}"


class ContextManager:

    def __init__(self, max_context_chars: int = 8000):
        self.max_context_chars = max_context_chars

    def prioritize_sources(
        self,
        sources: List[MinimalSource],
        repo_path: Path,
        question: str
    ) -> List[MinimalSource]:

        prioritized = []
        total_size = 0

        for source in sources:
            try:
                file_path = repo_path / source.file_path
                if not file_path.exists():
                    continue

                chunk_size = (
                    source.last_character_index - source.first_character_index
                )

                if total_size + chunk_size <= self.max_context_chars:
                    prioritized.append(source)
                    total_size += chunk_size
                else:
                    remaining = self.max_context_chars - total_size
                    if remaining > 500:
                        truncated = MinimalSource(
                            file_path=source.file_path,
                            first_character_index=source.first_character_index,
                            last_character_index=(
                                source.first_character_index + remaining
                            )
                        )
                        prioritized.append(truncated)
                    break

            except Exception:
                continue

        return prioritized
