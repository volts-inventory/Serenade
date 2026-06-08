"""Modal screen for creating a new engagement.

Calls ``orchestrator.services.engagements`` directly (no HTTP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from orchestrator.services import errors as svc_errors
from orchestrator.services import engagements as svc_engagements

if TYPE_CHECKING:
    from tui.app import SerenadeApp


_MODES = [
    ("recon_only (recon + enum only)", "recon_only"),
    ("assess (full pipeline, no exploitation)", "assess"),
    ("guided (exploit with per-gate approval)", "guided"),
    ("autonomous (exploit auto-executed)", "autonomous"),
]


class NewEngagementScreen(ModalScreen[dict | None]):
    """Returns the created engagement dict on success, ``None`` on cancel."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [("escape", "dismiss(None)", "Cancel")]

    def compose(self) -> ComposeResult:
        with Container(id="new-engagement-container"), Vertical(id="new-engagement-box"):
            yield Label("New engagement", classes="login-title")
            yield Static("[dim]LLM: (loading...)[/dim]", id="ne-llm-info")

            yield Label(
                "Auto mode (yes/no) — bundles autonomous + auto_start",
                classes="form-label",
            )
            yield Input(placeholder="no", value="no", id="ne-automode")

            yield Label("Name", classes="form-label")
            yield Input(placeholder="acme-q3-external", id="ne-name")

            yield Label("Scope CIDRs (comma-separated)", classes="form-label")
            yield Input(placeholder="10.0.0.0/24, 192.168.1.0/24", id="ne-cidrs")

            yield Label("Scope domains (comma-separated)", classes="form-label")
            yield Input(placeholder="example.com, api.example.com", id="ne-domains")

            yield Label("Mode", classes="form-label")
            yield Select(_MODES, value="guided", id="ne-mode", allow_blank=False)

            yield Label(
                "Token budget (e.g. 2000000; 0 = unlimited, only with local LLM)",
                classes="form-label",
            )
            yield Input(placeholder="2000000", value="2000000", id="ne-budget")

            yield Label("Wallclock hours", classes="form-label")
            yield Input(placeholder="8", value="8", id="ne-hours")

            yield Label(
                "Auto-start after creation (yes/no) — needs valid auth sha256",
                classes="form-label",
            )
            yield Input(placeholder="no", value="no", id="ne-autostart")

            yield Label(
                "Authorization sha256 (64 hex chars, optional at create-time)",
                classes="form-label",
            )
            yield Input(placeholder="", id="ne-auth")

            yield Static("", classes="field-error", id="ne-error")

            with Horizontal():
                yield Button("Create", variant="primary", id="ne-create")
                yield Button("Cancel", id="ne-cancel")

    async def on_mount(self) -> None:
        """Query LLM info so the operator can see whether the
        token budget actually caps a paid backend or is just a kill-
        switch for a local model."""
        info_widget = self.query_one("#ne-llm-info", Static)
        try:
            info = await svc_engagements.get_llm_info()
        except Exception:
            info_widget.update(
                "[dim]LLM: (info unavailable; budget treated as required)[/dim]"
            )
            return
        backend = info.get("backend", "?")
        model = info.get("model", "?")
        is_local = bool(info.get("is_local"))
        if is_local:
            info_widget.update(
                f"LLM: [green]{backend}[/green] [b]local[/b] · "
                f"model={model}   token_budget is OPTIONAL (0 = unlimited)"
            )
        else:
            info_widget.update(
                f"LLM: [yellow]{backend}[/yellow] hosted · model={model}"
                "   token_budget is REQUIRED"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ne-cancel":
            self.dismiss(None)
        elif event.button.id == "ne-create":
            self.app.run_worker(self._do_create(), exclusive=False)

    def _set_error(self, msg: str) -> None:
        self.query_one("#ne-error", Static).update(msg)

    async def _do_create(self) -> None:  # noqa: PLR0911
        name = self.query_one("#ne-name", Input).value.strip()
        cidrs = _split_csv(self.query_one("#ne-cidrs", Input).value)
        domains = _split_csv(self.query_one("#ne-domains", Input).value)
        mode = str(self.query_one("#ne-mode", Select).value)
        budget_s = self.query_one("#ne-budget", Input).value.strip() or "2000000"
        hours_s = self.query_one("#ne-hours", Input).value.strip() or "8"
        auth = self.query_one("#ne-auth", Input).value.strip()
        autostart_s = (
            self.query_one("#ne-autostart", Input).value.strip().lower()
        )
        auto_start = autostart_s in ("yes", "y", "true", "1")

        automode_s = (
            self.query_one("#ne-automode", Input).value.strip().lower()
        )
        if automode_s in ("yes", "y", "true", "1"):
            mode = "autonomous"
            auto_start = True

        if not name:
            self._set_error("name is required")
            return
        if not cidrs and not domains:
            self._set_error("scope: at least one cidr or domain required")
            return
        try:
            token_budget = int(budget_s)
            wallclock_hours = int(hours_s)
        except ValueError:
            self._set_error("token_budget and wallclock_hours must be integers")
            return

        if auto_start and not auth:
            self._set_error(
                "auto_start requires a valid 64-char hex authorization sha256"
            )
            return

        try:
            from worldstate.schema import Scope

            create_payload = svc_engagements.EngagementCreate(
                name=name,
                scope=Scope(
                    cidrs=cidrs,
                    domains=domains,
                    urls=[],
                    excluded_cidrs=[],
                    excluded_domains=[],
                    excluded_paths=[],
                ),
                objectives=[],
                mode=mode,
                token_budget=token_budget,
                wallclock_hours=wallclock_hours,
                authorization_artifact_sha256=auth,
                auto_start=auto_start,
            )
            row = await svc_engagements.create_engagement(create_payload)
            row_dict = row.model_dump(mode="json")
        except svc_errors.ServiceError as e:
            self._set_error(f"create failed: {e.message}")
            return
        except Exception as e:
            self._set_error(f"create failed: {e}")
            return
        self.dismiss(row_dict)


def _split_csv(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]
