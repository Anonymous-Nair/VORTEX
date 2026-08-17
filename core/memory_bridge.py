from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path


class VortexMemory:
    """Local, filesystem-backed VORTEX memory with lazy writes and a small file cache."""

    def __init__(
        self,
        vault: str | None = None,
        vortex_root: str = r"E:\VORTEX",
        cache_ttl_seconds: float = 2.0,
    ) -> None:
        self.vortex_root = Path(vortex_root).resolve()
        self.vault = Path(vault).resolve() if vault else self.vortex_root
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))

        self.identity = self.vault / "Identity"
        self.memory = self.vault / "Memory"
        self.long_term = self.memory / "Long_Term"
        self.conversations = self.memory / "Conversation_Dumps"
        self.decisions = self.memory / "Decisions"
        self.preferences = self.memory / "Preferences"
        self.knowledge = self.vault / "Knowledge"
        self.projects = self.vault / "Projects"
        self.tasks = self.vault / "Tasks"
        self.system = self.vault / "System"

        self._file_index: list[Path] = []
        self._file_index_expires = 0.0
        self._document_cache: dict[Path, tuple[int, int, list[str]]] = {}

    def _ensure_write_structure(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)

    def _files(self) -> list[Path]:
        now = time.monotonic()
        if now < self._file_index_expires:
            return self._file_index

        files: list[Path] = []
        try:
            files = [path for path in self.vault.rglob("*.md") if path.is_file()]
        except OSError:
            files = []

        self._file_index = files
        self._file_index_expires = now + self.cache_ttl_seconds
        return files

    def _blocks_for(self, path: Path) -> list[str]:
        try:
            stat = path.stat()
        except OSError:
            return []

        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._document_cache.get(path)
        if cached and cached[:2] == signature:
            return cached[2]

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        blocks = (
            self._conversation_blocks(text)
            if self._is_conversation_dump(path)
            else ([text.strip()] if text.strip() else [])
        )
        self._document_cache[path] = (signature[0], signature[1], blocks)
        return blocks

    def save_conversation(self, user_text: str, assistant_text: str) -> Path:
        now = datetime.now()
        path = self.conversations / f"{now:%Y-%m-%d}_Conversation_Dump.md"
        self._ensure_write_structure(path)
        timestamp = now.strftime("%H:%M:%S")
        if not path.exists():
            path.write_text(f"# VORTEX Conversation — {now:%Y-%m-%d}\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as file:
            file.write(
                f"## {timestamp}\n\n**Sir:**\n{user_text.strip()}\n\n"
                f"**VORTEX:**\n{assistant_text.strip()}\n\n---\n\n"
            )
        self._file_index_expires = 0.0
        self._document_cache.pop(path, None)
        return path

    def search(self, query: str, max_results: int = 8, min_score: int = 1) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        query_words = self._keywords(query)
        if not query_words:
            return []

        results: list[dict] = []
        for path in self._files():
            for block in self._blocks_for(path):
                result = self._score_memory_block(block, path, query_words, min_score)
                if result is not None:
                    results.append(result)

        results.sort(key=lambda item: (item["score"], item.get("matched_keywords", 0)), reverse=True)
        return results[:max_results]

    def _score_memory_block(self, block: str, path: Path, query_words: list[str], min_score: int) -> dict | None:
        content = block.strip()
        if not content:
            return None
        lowered = content.lower()
        matched_words = [word for word in query_words if word in lowered]
        if not matched_words:
            return None

        score = sum(lowered.count(word) for word in matched_words)
        normalized_query = self._normalize(query_words)
        normalized_content = self._normalize(self._keywords(content))
        if normalized_query and normalized_query in normalized_content:
            score += 8

        user_section = self._extract_user_section(content)
        if user_section:
            user_lower = user_section.lower()
            score += sum(user_lower.count(word) for word in matched_words) * 3

        if score < min_score:
            return None
        return {
            "path": str(path),
            "score": score,
            "matched_keywords": len(matched_words),
            "keywords": matched_words,
            "content": content,
        }

    def build_context(self, query: str, max_results: int = 5, max_chars: int = 12000) -> str:
        results = self.search(query, max_results=max_results)
        if not results:
            return ""
        sections: list[str] = []
        total_chars = 0
        for result in results:
            content = result["content"].strip()
            remaining = max_chars - total_chars
            if not content or remaining <= 0:
                break
            content = content[:remaining].rstrip()
            sections.append(
                "[MEMORY SOURCE]\n"
                f"{result['path']}\nRelevance score: {result['score']}\n\n{content}\n"
            )
            total_chars += len(content)
        if not sections:
            return ""
        return "\n\n===== VORTEX RELEVANT MEMORY =====\n" + "\n\n".join(sections) + "\n===== END VORTEX MEMORY =====\n"

    def save_long_term(self, title: str, content: str) -> Path:
        path = self.long_term / f"{self._safe_filename(title)}.md"
        self._write_markdown(path, title, content)
        return path

    def save_decision(self, title: str, content: str) -> Path:
        path = self.decisions / f"{self._safe_filename(title)}.md"
        self._write_markdown(path, title, content)
        return path

    def save_preference(self, title: str, content: str) -> Path:
        path = self.preferences / f"{self._safe_filename(title)}.md"
        self._write_markdown(path, title, content)
        return path

    def _write_markdown(self, path: Path, title: str, content: str) -> None:
        self._ensure_write_structure(path)
        path.write_text(f"# {title}\n\n{content.strip()}\n", encoding="utf-8")
        self._file_index_expires = 0.0
        self._document_cache.pop(path, None)

    @staticmethod
    def _is_conversation_dump(path: Path) -> bool:
        return "Conversation_Dumps" in path.parts

    @staticmethod
    def _conversation_blocks(text: str) -> list[str]:
        matches = list(re.finditer(r"(?m)^##\s+\d{2}:\d{2}:\d{2}\s*$", text))
        if not matches:
            return [text.strip()] if text.strip() else []
        blocks: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.start():end].strip()
            block = re.sub(r"\n---\s*$", "", block).strip()
            if block:
                blocks.append(block)
        return blocks

    @staticmethod
    def _extract_user_section(content: str) -> str:
        match = re.search(r"\*\*Sir:\*\*\s*(.*?)(?=\n\s*\*\*VORTEX:\*\*|\Z)", content, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _keywords(text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        stopwords = {
            "the", "and", "for", "that", "this", "with", "what", "when", "where", "how", "why",
            "are", "was", "were", "you", "your", "can", "could", "would", "should", "does", "did",
            "have", "has", "from", "into", "about", "please", "vortex", "sir", "hey", "hi", "hello",
            "hiya", "there", "yeah", "yep", "nope", "okay", "thanks", "thank", "welcome", "tell", "told",
            "said", "say", "remember", "recall", "know", "knew", "today", "yesterday", "recently", "thing", "things",
        }
        return list(dict.fromkeys(word for word in words if len(word) >= 3 and word not in stopwords))

    @staticmethod
    def _normalize(words: list[str]) -> str:
        return " ".join(word.lower().strip() for word in words if word.strip())

    @staticmethod
    def _safe_filename(title: str) -> str:
        title = re.sub(r'[<>:"/\\|?*]', "_", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title[:120] or "Untitled"
