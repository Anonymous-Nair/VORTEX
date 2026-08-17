from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


class VortexMemory:
    """
    Local memory layer for VORTEX.

    Uses the Obsidian vault directly through the filesystem.
    No external API, database, or cloud service required.

    Memory retrieval is block-based:
        - Conversation dumps are split into individual conversation blocks.
        - Matching is performed against individual blocks rather than
          entire daily files.
        - Relevant blocks are ranked by keyword matches.
        - The returned context contains only the most relevant material.
    """

    def __init__(
        self,
        vault: str = r"E:\DedSec\VORTEX",
    ) -> None:
        self.vault = Path(vault)

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

        self._ensure_structure()

    # =========================================================
    # DIRECTORY STRUCTURE
    # =========================================================

    def _ensure_structure(self) -> None:
        folders = [
            self.identity,
            self.memory,
            self.long_term,
            self.conversations,
            self.decisions,
            self.preferences,
            self.knowledge,
            self.projects,
            self.tasks,
            self.system,
        ]

        for folder in folders:
            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

    # =========================================================
    # DAILY CONVERSATION MEMORY
    # =========================================================

    def save_conversation(
        self,
        user_text: str,
        assistant_text: str,
    ) -> Path:
        now = datetime.now()

        filename = (
            f"{now:%Y-%m-%d}_Conversation_Dump.md"
        )

        path = self.conversations / filename

        timestamp = now.strftime("%H:%M:%S")

        if not path.exists():
            path.write_text(
                f"# VORTEX Conversation — "
                f"{now:%Y-%m-%d}\n\n",
                encoding="utf-8",
            )

        with path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                f"## {timestamp}\n\n"
                f"**Sir:**\n"
                f"{user_text.strip()}\n\n"
                f"**VORTEX:**\n"
                f"{assistant_text.strip()}\n\n"
                f"---\n\n"
            )

        return path

    # =========================================================
    # SEARCH MEMORY
    # =========================================================

    def search(
        self,
        query: str,
        max_results: int = 8,
        min_score: int = 1,
    ) -> list[dict]:
        """
        Search VORTEX memory and return relevant memory blocks.

        Conversation dumps are searched block-by-block instead of
        treating an entire daily dump as one document.

        This prevents unrelated conversations from contaminating
        the result simply because they happen to exist in the same
        daily Markdown file.
        """

        query = query.strip()

        if not query:
            return []

        query_words = self._keywords(query)

        if not query_words:
            return []

        results: list[dict] = []

        for path in self.vault.rglob("*.md"):

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

            except Exception:
                continue

            # -------------------------------------------------
            # Conversation dumps
            # -------------------------------------------------

            if self._is_conversation_dump(path):
                blocks = self._conversation_blocks(text)

                for block in blocks:
                    result = self._score_memory_block(
                        block=block,
                        path=path,
                        query_words=query_words,
                        min_score=min_score,
                    )

                    if result is not None:
                        results.append(result)

            # -------------------------------------------------
            # Other Markdown memory
            # -------------------------------------------------

            else:
                result = self._score_memory_block(
                    block=text,
                    path=path,
                    query_words=query_words,
                    min_score=min_score,
                )

                if result is not None:
                    results.append(result)

        # Highest relevance first.
        results.sort(
            key=lambda item: (
                item["score"],
                item.get("matched_keywords", 0),
            ),
            reverse=True,
        )

        return results[:max_results]

    # =========================================================
    # MEMORY BLOCK SCORING
    # =========================================================

    def _score_memory_block(
        self,
        block: str,
        path: Path,
        query_words: list[str],
        min_score: int,
    ) -> dict | None:
        """
        Score one isolated memory block.

        Returns None when the block is not relevant.
        """

        content = block.strip()

        if not content:
            return None

        lowered = content.lower()

        matched_words: list[str] = []

        for word in query_words:
            if word in lowered:
                matched_words.append(word)

        if not matched_words:
            return None

        # -----------------------------------------------------
        # Base score
        # -----------------------------------------------------

        score = sum(
            lowered.count(word)
            for word in matched_words
        )

        # -----------------------------------------------------
        # Exact phrase bonus
        # -----------------------------------------------------

        normalized_query = self._normalize(query_words)

        normalized_content = self._normalize(
            self._keywords(content)
        )

        if normalized_query and normalized_query in normalized_content:
            score += 8

        # -----------------------------------------------------
        # User-message bonus
        #
        # A memory block containing the user's statement is more
        # valuable than a block where the assistant merely repeats
        # the keyword.
        # -----------------------------------------------------

        user_section = self._extract_user_section(
            content
        )

        if user_section:
            user_lower = user_section.lower()

            user_matches = sum(
                user_lower.count(word)
                for word in matched_words
            )

            score += user_matches * 3

        # -----------------------------------------------------
        # Minimum relevance threshold
        # -----------------------------------------------------

        if score < min_score:
            return None

        return {
            "path": str(path),
            "score": score,
            "matched_keywords": len(
                matched_words
            ),
            "keywords": matched_words,
            "content": content,
        }

    # =========================================================
    # BUILD MEMORY CONTEXT FOR LLM
    # =========================================================

    def build_context(
        self,
        query: str,
        max_results: int = 5,
        max_chars: int = 12000,
    ) -> str:
        """
        Build compact, relevant memory context for the LLM.

        Only the highest-ranked memory blocks are included.
        """

        results = self.search(
            query,
            max_results=max_results,
        )

        if not results:
            return ""

        sections: list[str] = []
        total_chars = 0

        for result in results:

            content = result["content"].strip()

            if not content:
                continue

            remaining = max_chars - total_chars

            if remaining <= 0:
                break

            if len(content) > remaining:
                content = content[:remaining].rstrip()

            sections.append(
                "[MEMORY SOURCE]\n"
                f"{result['path']}\n"
                f"Relevance score: {result['score']}\n\n"
                f"{content}\n"
            )

            total_chars += len(content)

        if not sections:
            return ""

        return (
            "\n\n"
            "===== VORTEX RELEVANT MEMORY =====\n"
            + "\n\n".join(sections)
            + "\n===== END VORTEX MEMORY =====\n"
        )

    # =========================================================
    # STORE LONG-TERM MEMORY
    # =========================================================

    def save_long_term(
        self,
        title: str,
        content: str,
    ) -> Path:
        safe_title = self._safe_filename(
            title
        )

        path = (
            self.long_term
            / f"{safe_title}.md"
        )

        path.write_text(
            f"# {title}\n\n"
            f"{content.strip()}\n",
            encoding="utf-8",
        )

        return path

    # =========================================================
    # STORE DECISION
    # =========================================================

    def save_decision(
        self,
        title: str,
        content: str,
    ) -> Path:
        safe_title = self._safe_filename(
            title
        )

        path = (
            self.decisions
            / f"{safe_title}.md"
        )

        path.write_text(
            f"# {title}\n\n"
            f"{content.strip()}\n",
            encoding="utf-8",
        )

        return path

    # =========================================================
    # STORE PREFERENCE
    # =========================================================

    def save_preference(
        self,
        title: str,
        content: str,
    ) -> Path:
        safe_title = self._safe_filename(
            title
        )

        path = (
            self.preferences
            / f"{safe_title}.md"
        )

        path.write_text(
            f"# {title}\n\n"
            f"{content.strip()}\n",
            encoding="utf-8",
        )

        return path

    # =========================================================
    # CONVERSATION DETECTION
    # =========================================================

    @staticmethod
    def _is_conversation_dump(
        path: Path,
    ) -> bool:
        return (
            "Conversation_Dumps"
            in path.parts
        )

    # =========================================================
    # CONVERSATION BLOCK PARSER
    # =========================================================

    @staticmethod
    def _conversation_blocks(
        text: str,
    ) -> list[str]:
        """
        Split a conversation dump into individual turns.

        Example:

            ## 13:30:07

            **Sir:**
            What did I tell you about mango?

            **VORTEX:**
            Mango is the king of fruits.

        becomes one isolated memory block.
        """

        matches = list(
            re.finditer(
                r"(?m)^##\s+\d{2}:\d{2}:\d{2}\s*$",
                text,
            )
        )

        if not matches:
            return [
                text.strip()
            ] if text.strip() else []

        blocks: list[str] = []

        for index, match in enumerate(matches):

            start = match.start()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(text)

            block = text[
                start:end
            ].strip()

            # Remove trailing separator.
            block = re.sub(
                r"\n---\s*$",
                "",
                block,
            ).strip()

            if block:
                blocks.append(block)

        return blocks

    # =========================================================
    # USER SECTION EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_user_section(
        content: str,
    ) -> str:
        """
        Extract the user's portion from a conversation block.
        """

        match = re.search(
            r"\*\*Sir:\*\*\s*(.*?)(?=\n\s*\*\*VORTEX:\*\*|\Z)",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if not match:
            return ""

        return match.group(1).strip()

    # =========================================================
    # KEYWORD EXTRACTION
    # =========================================================

    @staticmethod
    def _keywords(
        text: str,
    ) -> list[str]:
        words = re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )

        stopwords = {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "what",
            "when",
            "where",
            "how",
            "why",
            "are",
            "was",
            "were",
            "you",
            "your",
            "can",
            "could",
            "would",
            "should",
            "does",
            "did",
            "have",
            "has",
            "from",
            "into",
            "about",
            "please",
            "vortex",
            "sir",
            "hey",
            "hi",
            "hello",
            "hiya",
            "there",
            "yeah",
            "yep",
            "nope",
            "okay",
            "thanks",
            "thank",
            "welcome",
            "tell",
            "told",
            "said",
            "say",
            "remember",
            "recall",
            "know",
            "knew",
            "today",
            "yesterday",
            "recently",
            "thing",
            "things",
        }

        result: list[str] = []

        for word in words:

            if len(word) < 3:
                continue

            if word in stopwords:
                continue

            if word not in result:
                result.append(word)

        return result

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize(
        words: list[str],
    ) -> str:
        return " ".join(
            word.lower().strip()
            for word in words
            if word.strip()
        )

    # =========================================================
    # SAFE FILENAME
    # =========================================================

    @staticmethod
    def _safe_filename(
        title: str,
    ) -> str:
        title = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            title,
        )

        title = re.sub(
            r"\s+",
            " ",
            title,
        ).strip()

        return (
            title[:120]
            or "Untitled"
        )