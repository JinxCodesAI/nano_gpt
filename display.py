from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple


class DiffusionDisplay:
    """Handle coloration and display of diffusion sampling iterations."""

    GREEN = "\033[92m"
    ORANGE = "\033[38;5;208m"
    RED_BACKGROUND = "\033[41m"
    WHITE_BACKGROUND = "\033[47m"
    RESET = "\033[0m"
    SEPARATOR = "=" * 122

    def __init__(self, decode: Callable[[Sequence[int]], str]):
        self._decode = decode
        self._prev_decoded: Optional[str] = None
        self._token_index_map: List[Optional[int]] = []
        self._pending_deletion_display: Optional[str] = None

    def reset(self) -> None:
        """Clear any cached state between samples."""
        self._prev_decoded = None
        self._token_index_map = []
        self._pending_deletion_display = None

    def start_iteration(self, max_token_pos: int) -> None:
        """Initialise tracking structures for a new diffusion iteration."""
        self._token_index_map = list(range(max_token_pos))
        self._pending_deletion_display = None

    def register_insertions(self, applied_insertions: Sequence[int]) -> None:
        """Track how freshly inserted tokens shift later comparisons."""
        offset = 0
        for idx in applied_insertions:
            pos = idx + offset
            if 0 <= pos <= len(self._token_index_map):
                self._token_index_map.insert(pos, None)
            offset += 1

    def capture_pre_deletion(
        self,
        pre_deletion_tokens: Optional[Sequence[int]],
        applied_deletions: Sequence[int],
    ) -> None:
        """Store a highlighted view of tokens scheduled for deletion."""
        if not pre_deletion_tokens or not applied_deletions:
            self._pending_deletion_display = None
            return

        pre_decoded = self._decode(pre_deletion_tokens)
        deletion_set = set(applied_deletions)
        highlighted = []
        for idx, char in enumerate(pre_decoded):
            if idx in deletion_set:
                highlighted.append(f"{self.RED_BACKGROUND}{char}{self.RESET}")
            else:
                highlighted.append(char)
        self._pending_deletion_display = "".join(highlighted)

    def register_deletions(self, applied_deletions: Sequence[int]) -> None:
        """Remove deleted slots from the provenance map."""
        for idx in sorted(applied_deletions, reverse=True):
            if 0 <= idx < len(self._token_index_map):
                self._token_index_map.pop(idx)

    def render(self, tokens: Sequence[int]) -> Tuple[Optional[str], str, int, int]:
        """Return the coloured output and change counts for the iteration."""
        decoded = self._decode(tokens)
        colored_chars = []
        green_count = 0
        orange_count = 0

        for idx, char in enumerate(decoded):
            source_idx = (
                self._token_index_map[idx] if idx < len(self._token_index_map) else None
            )
            if source_idx is None:
                orange_count += 1
                colored_chars.append(
                    f"{self.WHITE_BACKGROUND}{self.ORANGE}{char}{self.RESET}"
                )
                continue

            same_as_prev = (
                self._prev_decoded is not None
                and source_idx < len(self._prev_decoded)
                and char == self._prev_decoded[source_idx]
            )
            if same_as_prev:
                green_count += 1
                colored_chars.append(f"{self.GREEN}{char}{self.RESET}")
            else:
                orange_count += 1
                colored_chars.append(f"{self.ORANGE}{char}{self.RESET}")

        output = "".join(colored_chars)
        deletion_display = self._pending_deletion_display
        self._prev_decoded = decoded
        return deletion_display, output, green_count, orange_count

    def emit_iteration(
        self,
        *,
        iteration: int,
        sample_index: int,
        mean_log_prob: float,
        insert_count: int,
        delete_count: int,
        tokens: Sequence[int],
    ) -> None:
        """Print the coloured iteration summary."""
        deletion_display, colored_output, green_count, orange_count = self.render(tokens)
        total_colored = green_count + orange_count
        changed_summary = (
            f"{orange_count}/{total_colored}" if total_colored else "0/0"
        )
        print(
            f"Iteration {iteration}, sample {sample_index} | "
            f"mean log prob: {mean_log_prob:.4f} | "
            f"changed {changed_summary} | "
            f"ins {insert_count} | del {delete_count}"
        )
        if deletion_display is not None:
            print(f"Before deletion: {deletion_display}")
        print(colored_output)
        print(self.SEPARATOR)

    def emit_final_tokens(self, tokens: Sequence[int]) -> None:
        """Print the final decoded sequence for the sample."""
        text = self._decode(tokens)
        print(text)
        print("---------------")


__all__ = ["DiffusionDisplay"]
