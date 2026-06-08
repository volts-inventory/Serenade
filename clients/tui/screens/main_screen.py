"""Main screen — sidebar + tabbed right pane.

Orchestrates all the widgets, owns the periodic poll for engagements/cost/
approvals, and routes user actions (lifecycle, approvals, operator-exec,
report downloads) through direct service calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, TabbedContent, TabPane

from orchestrator.services import errors as svc_errors
from orchestrator.services import engagements as svc_engagements
from orchestrator.services import cost as svc_cost
from orchestrator.services import findings as svc_findings
from orchestrator.services import reports as svc_reports
from orchestrator.services import operator_exec as svc_operator_exec
from tui.screens.finding_detail import FindingDetailScreen
from tui.screens.new_engagement import NewEngagementScreen
from tui.widgets.approvals_panel import ApprovalDecisionRequested, ApprovalsPanel
from tui.widgets.engagement_list import EngagementList, EngagementSelected
from tui.widgets.event_stream import EventStreamPanel
from tui.widgets.findings_panel import FindingActivated, FindingsPanel, HostsPanel
from tui.widgets.operator_exec_panel import OperatorExecPanel, OperatorExecRequested
from tui.widgets.overview_panel import EngagementLifecycleAction, OverviewPanel
from tui.widgets.report_panel import ReportDownloadRequested, ReportPanel
from tui.widgets.status_bar import StatusBar

if TYPE_CHECKING:
    from tui.app import SerenadeApp


_TAB_IDS = [
    "tab-overview",
    "tab-events",
    "tab-approvals",
    "tab-findings",
    "tab-hosts",
    "tab-opexec",
    "tab-report",
]


class MainScreen(Screen):
    """Top-level operator screen — sidebar, tabs, footer."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("q", "app.quit", "Quit"),
        ("n", "new_engagement", "New engagement"),
        ("r", "refresh", "Refresh"),
        ("p", "cycle_mode", "Cycle mode"),
        ("1", "switch_tab('tab-overview')", "Overview"),
        ("2", "switch_tab('tab-events')", "Events"),
        ("3", "switch_tab('tab-approvals')", "Approvals"),
        ("4", "switch_tab('tab-findings')", "Findings"),
        ("5", "switch_tab('tab-hosts')", "Hosts"),
        ("6", "switch_tab('tab-opexec')", "Op-Exec"),
        ("7", "switch_tab('tab-report')", "Report"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._engagement_id: str | None = None
        self._engagement: dict[str, Any] | None = None
        # Refresh intervals (seconds)
        self._approval_poll_secs = 5.0
        self._cost_poll_secs = 30.0
        self._eng_poll_secs = 15.0

    # ------------------------------------------------------------ compose
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-grid"):
            yield EngagementList()
            with TabbedContent(id="right-pane", initial=_TAB_IDS[0]):
                with TabPane("Overview", id=_TAB_IDS[0]):
                    yield OverviewPanel()
                with TabPane("Events", id=_TAB_IDS[1]):
                    yield EventStreamPanel()
                with TabPane("Approvals", id=_TAB_IDS[2]):
                    yield ApprovalsPanel()
                with TabPane("Findings", id=_TAB_IDS[3]):
                    yield FindingsPanel()
                with TabPane("Hosts", id=_TAB_IDS[4]):
                    yield HostsPanel()
                with TabPane("Op-Exec", id=_TAB_IDS[5]):
                    yield OperatorExecPanel()
                with TabPane("Report", id=_TAB_IDS[6]):
                    yield ReportPanel()
        yield StatusBar()
        yield Footer()

    # ------------------------------------------------------------- mount
    async def on_mount(self) -> None:
        app: SerenadeApp = self.app  # type: ignore[assignment]
        sb = self.query_one(StatusBar)
        sb.set_connection("native", "local", "local")
        await self._refresh_engagements()
        # Set up periodic polls. set_interval returns a Timer.
        self.set_interval(self._eng_poll_secs, self._refresh_engagements)
        self.set_interval(self._approval_poll_secs, self._refresh_approvals)
        self.set_interval(self._cost_poll_secs, self._refresh_cost)

    # ------------------------------------------------------------ actions
    def action_switch_tab(self, tab_id: str) -> None:
        try:
            tabs = self.query_one("#right-pane", TabbedContent)
            tabs.active = tab_id
        except Exception:
            pass

    async def action_new_engagement(self) -> None:
        # ModalScreen returns either a dict (created) or None (cancel)
        result = await self.app.push_screen_wait(NewEngagementScreen())
        if isinstance(result, dict):
            await self._refresh_engagements()

    async def action_cycle_mode(self) -> None:
        """Cycle the selected engagement's mode through
        recon_only -> assess -> guided -> autonomous -> recon_only.

        Mode is the single config axis — allowed risk classes and approval
        behaviour are derived from it (guided gates exploits for approval;
        autonomous auto-executes)."""
        if self._engagement_id is None:
            self.notify("no engagement selected", severity="warning", timeout=2)
            return
        eng = self._engagement or {}
        if str(eng.get("status", "")) == "running":
            self.notify(
                "mode cannot change while running — pause first",
                severity="warning",
                timeout=3,
            )
            return
        nxt_map = {
            "recon_only": "assess",
            "assess": "guided",
            "guided": "autonomous",
            "autonomous": "recon_only",
        }
        nxt = nxt_map.get(str(eng.get("mode", "guided")), "guided")
        try:
            update = svc_engagements.EngagementUpdate(mode=nxt)
            row = await svc_engagements.update_engagement(
                UUID(self._engagement_id), update
            )
            row_dict = row.model_dump(mode="json")
        except svc_errors.ServiceError as e:
            self.notify(f"mode: {e.message}", severity="error", timeout=3)
            return
        except Exception as e:
            self.notify(f"mode: {e}", severity="error", timeout=3)
            return
        self._engagement = row_dict
        self.query_one(OverviewPanel).set_engagement(row_dict)
        self.notify(f"mode -> {nxt}", severity="information", timeout=2)

    async def action_refresh(self) -> None:
        await self._refresh_engagements()
        if self._engagement_id is not None:
            await self._refresh_selected()

    # ------------------------------------------------------- refresh helpers
    async def _refresh_engagements(self) -> None:
        try:
            rows_typed = await svc_engagements.list_engagements()
            rows = [r.model_dump(mode="json") for r in rows_typed]
        except svc_errors.ServiceError as e:
            self.notify(f"engagements: {e.message}", severity="error", timeout=3)
            return
        except Exception as e:
            self.notify(f"engagements: {e}", severity="error", timeout=3)
            return
        lst = self.query_one(EngagementList)
        lst.update_engagements(rows)
        # If we don't have one selected yet, pick the last-used or the first.
        if self._engagement_id is None:
            app: SerenadeApp = self.app  # type: ignore[assignment]
            last = app.cfg.last_engagement_id
            chosen: str | None = None
            if last and any(str(r.get("id")) == last for r in rows):
                chosen = last
            elif rows:
                chosen = str(rows[0].get("id"))
            if chosen is not None:
                await self._select_engagement(chosen)

    async def _refresh_selected(self) -> None:
        if self._engagement_id is None:
            return
        await self._refresh_engagement_detail()
        await self._refresh_findings()
        await self._refresh_hosts()
        await self._refresh_approvals()
        await self._refresh_cost()

    async def _refresh_engagement_detail(self) -> None:
        if self._engagement_id is None:
            return
        try:
            row = await svc_engagements.get_engagement(UUID(self._engagement_id))
            row_dict = row.model_dump(mode="json")
        except svc_errors.ServiceError as e:
            self.notify(f"engagement: {e.message}", severity="warning", timeout=3)
            return
        except Exception:
            return
        self._engagement = row_dict
        self.query_one(OverviewPanel).set_engagement(row_dict)

    async def _refresh_findings(self) -> None:
        if self._engagement_id is None:
            return
        try:
            data = await svc_findings.get_findings(UUID(self._engagement_id))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        self.query_one(FindingsPanel).update_findings(data)
        self.query_one(ReportPanel).update_findings(data)

    async def _refresh_hosts(self) -> None:
        if self._engagement_id is None:
            return
        try:
            rows = await svc_findings.get_hosts(UUID(self._engagement_id))
        except Exception:
            return
        if not isinstance(rows, list):
            return
        self.query_one(HostsPanel).update_hosts(rows)

    async def _refresh_approvals(self) -> None:
        if self._engagement_id is None:
            return
        try:
            rows_typed = await svc_engagements.list_approvals(UUID(self._engagement_id))
            rows = [r.model_dump(mode="json") for r in rows_typed]
        except Exception:
            return
        if not isinstance(rows, list):
            return
        self.query_one(ApprovalsPanel).update_approvals(rows)
        pending = sum(1 for r in rows if not r.get("decision"))
        self.query_one(StatusBar).set_pending(pending)

    async def _refresh_cost(self) -> None:
        if self._engagement_id is None:
            return
        try:
            data = await svc_cost.get_cost_actual(UUID(self._engagement_id))
        except Exception:
            return
        self.query_one(OverviewPanel).set_cost(data)
        self.query_one(StatusBar).set_cost(float(data.get("cost_usd", 0.0)))

    async def _select_engagement(self, engagement_id: str) -> None:
        app: SerenadeApp = self.app  # type: ignore[assignment]
        self._engagement_id = engagement_id
        app.cfg.last_engagement_id = engagement_id
        self.query_one(StatusBar).set_engagement(engagement_id)
        # Attach in-process event stream
        self.query_one(EventStreamPanel).attach(engagement_id)
        await self._refresh_selected()

    # ----------------------------------------------------------- messages
    async def on_engagement_selected(self, message: EngagementSelected) -> None:
        await self._select_engagement(message.engagement_id)

    async def on_engagement_lifecycle_action(
        self, message: EngagementLifecycleAction
    ) -> None:
        if self._engagement_id is None:
            return
        try:
            eid = UUID(self._engagement_id)
            if message.action == "start":
                await svc_engagements.start_engagement(eid)
            elif message.action == "pause":
                await svc_engagements.pause_engagement(eid)
            elif message.action == "resume":
                await svc_engagements.resume_engagement(eid)
        except svc_errors.ServiceError as e:
            self.notify(f"{message.action} failed: {e.message}", severity="error")
            return
        except Exception as e:
            self.notify(f"{message.action} failed: {e}", severity="error")
            return
        self.notify(f"{message.action} sent", severity="information", timeout=2)
        await self._refresh_engagement_detail()

    async def on_approval_decision_requested(
        self, message: ApprovalDecisionRequested
    ) -> None:
        if self._engagement_id is None:
            return
        try:
            eid = UUID(self._engagement_id)
            if message.bulk:
                payload = svc_engagements.BulkApprovalDecision(
                    decision=message.decision, kinds=[message.kind]
                )
                result = await svc_engagements.bulk_decide_approvals(eid, payload)
                result_dict = result.model_dump(mode="json")
                self.notify(
                    f"bulk {message.decision} -> {result_dict.get('decided', 0)} decided",
                    severity="information",
                )
            else:
                payload_single = svc_engagements.ApprovalDecision(
                    decision=message.decision
                )
                await svc_engagements.decide_approval(
                    eid,
                    UUID(message.approval_id),
                    payload_single,
                )
                self.notify(
                    f"approval {message.approval_id[:8]} {message.decision}d",
                    severity="information",
                )
        except svc_errors.ServiceError as e:
            self.notify(f"decide failed: {e.message}", severity="error")
            return
        except Exception as e:
            self.notify(f"decide failed: {e}", severity="error")
            return
        _ = message.notes
        await self._refresh_approvals()

    async def on_finding_activated(self, message: FindingActivated) -> None:
        await self.app.push_screen(FindingDetailScreen(message.finding))

    async def on_operator_exec_requested(self, message: OperatorExecRequested) -> None:
        panel = self.query_one(OperatorExecPanel)
        if self._engagement_id is None:
            panel.show_output("no engagement selected")
            return
        try:
            req = svc_operator_exec.OperatorExecRequest(**message.payload)
            res = await svc_operator_exec.run_operator_exec(
                UUID(self._engagement_id), req
            )
        except svc_errors.ServiceError as e:
            panel.show_output(f"error: {e.message}")
            return
        except Exception as e:
            panel.show_output(f"error: {e}")
            return
        panel.show_output(json.dumps(res, indent=2, default=str))

    async def on_report_download_requested(
        self, message: ReportDownloadRequested
    ) -> None:
        app: SerenadeApp = self.app  # type: ignore[assignment]
        panel = self.query_one(ReportPanel)
        if self._engagement_id is None:
            panel.show_result("no engagement selected")
            return
        try:
            data, _ctype = await svc_reports.render_report(
                UUID(self._engagement_id), message.fmt
            )
        except svc_errors.PermanentFailure:
            if message.fmt == "pdf":
                panel.show_result(
                    "[yellow]PDF unavailable: WeasyPrint not installed[/yellow]"
                )
            else:
                panel.show_result("[red]render failed: service unavailable[/red]")
            return
        except svc_errors.ServiceError as e:
            panel.show_result(f"[red]download failed: {e.message}[/red]")
            return
        except Exception as e:
            panel.show_result(f"[red]download failed: {e}[/red]")
            return
        ext_map = {
            "html": "html",
            "markdown": "md",
            "json": "json",
            "pdf": "pdf",
            "sarif": "sarif.json",
        }
        ext = ext_map.get(message.fmt, "bin")
        out_dir = Path(app.cfg.download_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"report-{self._engagement_id}.{ext}"
        out_path.write_bytes(data)
        panel.show_result(f"[green]wrote {out_path}[/green]")
