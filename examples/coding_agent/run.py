import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_NAME = "deepseek/deepseek-chat"
_stream_state = {"active": False}


def require_api_key() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")


def print_event(event) -> None:
    if event.type == "stream.token":
        token = event.payload.get("token", "")
        if token:
            sys.stdout.write(token)
            sys.stdout.flush()
            _stream_state["active"] = True
        return
    if _stream_state["active"]:
        print()
        _stream_state["active"] = False
    if event.type == "assistant.message":
        content = event.payload.get("content", "")
        tool_calls = event.payload.get("tool_calls", [])
        if content:
            print(f"assistant.message [{event.source}] {content}")
        if tool_calls:
            names = ", ".join(call.get("name", "") for call in tool_calls if call.get("name"))
            if names:
                print(f"assistant.tools [{event.source}] {names}")
            else:
                print(f"assistant.tools [{event.source}]")
        if not content and not tool_calls:
            print(f"assistant.message [{event.source}]")
        return
    payload = event.payload
    tail = ""
    if payload:
        tail = f" {payload}"
    print(f"{event.type} [{event.source}]" + tail)


def main() -> None:
    require_api_key()

    from blackgeorge import Desk, Job, Worker, Workforce
    from blackgeorge.workflow import Parallel, Step
    from examples.coding_agent.schema import ChangeReport
    from examples.coding_agent.tools import (
        ask_user,
        list_files,
        modified_files,
        read_file,
        restore_modified_files,
        write_file,
    )

    storage_dir = str(Path(__file__).resolve().parent / ".blackgeorge")
    stream_enabled = os.getenv("BLACKGEORGE_STREAM", "1") == "1"
    desk = Desk(
        model=MODEL_NAME,
        storage_dir=storage_dir,
        max_iterations=35,
        stream=stream_enabled,
    )
    keep_changes = os.getenv("PRESERVE_EXAMPLE_CHANGES") == "1"

    for event_type in [
        "run.started",
        "run.paused",
        "run.resumed",
        "run.completed",
        "run.failed",
        "workforce.started",
        "workforce.completed",
        "worker.started",
        "worker.completed",
        "worker.failed",
        "tool.started",
        "tool.completed",
        "tool.failed",
        "tool.confirmation_requested",
        "tool.user_input_requested",
        "assistant.message",
        "stream.token",
    ]:
        desk.event_bus.subscribe(event_type, print_event)

    manager = Worker(
        name="Manager",
        model=MODEL_NAME,
        instructions=(
            "You select the best worker for the task. "
            "Always choose Coder unless the task is only a summary."
        ),
    )

    coder = Worker(
        name="Coder",
        model=MODEL_NAME,
        tools=[list_files, read_file, write_file, ask_user],
        instructions=(
            "You are a coding agent working inside a small project. "
            "Use list_files and read_file to inspect the project. "
            "Spec is in spec.txt. "
            "Before deciding behavior for divide by zero or empty averages, "
            "call ask_user with a specific question in the question field. "
            "Use write_file to apply changes."
        ),
    )

    reviewer = Worker(
        name="Reviewer",
        model=MODEL_NAME,
        instructions=(
            "You summarize the changes and provide a structured report. "
            "Only respond with the schema fields."
        ),
    )

    narrator = Worker(
        name="Narrator",
        model=MODEL_NAME,
        instructions=("You produce a concise human-readable summary of the changes."),
    )

    workforce = Workforce(
        [coder, reviewer],
        mode="managed",
        name="coding_team",
        manager=manager,
    )

    job = Job(
        input={
            "task": "Fix calculator behavior and update tests.",
            "project": "Use tools to inspect the project files.",
            "requirements": [
                "Confirm divide-by-zero behavior with ask_user.",
                "Confirm empty-average behavior with ask_user.",
                "Apply changes using write_file.",
            ],
        },
        expected_output="Updated project files with consistent behavior.",
    )

    try:
        report = desk.run(workforce, job, stream=stream_enabled)

        while report.status == "paused" and report.pending_action is not None:
            action = report.pending_action
            if action.type == "confirmation":
                decision = input(f"{action.prompt} [y/n]: ").strip().lower() in {"y", "yes"}
            else:
                decision = input(f"{action.prompt}: ").strip()
            report = desk.resume(report, decision, stream=stream_enabled)

        if report.status != "completed":
            print("Run did not complete")
            print(report.errors)
            return

        flow_job = Job(
            input={
                "source_report": report.content or "",
                "notes": "Generate a summary and a structured change report.",
            }
        )

        def review_job(context):
            return Job(
                input={
                    "summary_source": context.job.input,
                    "instruction": "Return ChangeReport for the coding changes.",
                },
                response_schema=ChangeReport,
            )

        def narrative_job(context):
            return Job(
                input={
                    "summary_source": context.job.input,
                    "instruction": "Write a short plain-text summary.",
                }
            )

        flow = desk.flow(
            [
                Step(reviewer, job_builder=review_job),
                Parallel(
                    Step(narrator, job_builder=narrative_job),
                    Step(reviewer, job_builder=review_job),
                ),
            ]
        )

        flow_report = flow.run(flow_job)

        print("Final report status:", flow_report.status)
        print("Final report content:\n", flow_report.content)
        print("Run store path:", desk.db_path)
        print("Events stored:", len(desk.run_store.get_events(report.run_id)))
    finally:
        if not keep_changes:
            changed = modified_files()
            if changed:
                restored = restore_modified_files()
                print("Reverted example files:", ", ".join(restored))


if __name__ == "__main__":
    main()
