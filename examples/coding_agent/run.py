import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_NAME = "deepseek/deepseek-chat"
_stream_state = {"active": False}
_tool_state = {
    "active": set(),
    "started_at": {},
    "parallel_batches": 0,
}


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
    pretty_type = event.type.replace(".", " ").upper()
    if event.type == "tool.started":
        tool_call_id = event.payload.get("tool_call_id")
        if tool_call_id:
            _tool_state["active"].add(tool_call_id)
            _tool_state["started_at"][tool_call_id] = time.perf_counter()
            if len(_tool_state["active"]) == 2:
                _tool_state["parallel_batches"] += 1
                active_list = ", ".join(sorted(_tool_state["active"]))
                print(f"[PARALLEL] tools in flight: {active_list}")
    if event.type in {"tool.completed", "tool.failed"}:
        tool_call_id = event.payload.get("tool_call_id")
        if tool_call_id:
            started = _tool_state["started_at"].pop(tool_call_id, None)
            _tool_state["active"].discard(tool_call_id)
            if started is not None:
                elapsed = time.perf_counter() - started
                print(f"[TOOL TIMING] {tool_call_id} {elapsed:.3f}s")
    if event.type == "assistant.message":
        content = event.payload.get("content", "")
        tool_calls = event.payload.get("tool_calls", [])
        if content:
            print(f"[{pretty_type}] {event.source}: {content}")
        if tool_calls:
            names = ", ".join(call.get("name", "") for call in tool_calls if call.get("name"))
            if names:
                print(f"[ASSISTANT TOOLS] {event.source}: {names}")
            else:
                print(f"[ASSISTANT TOOLS] {event.source}")
        if not content and not tool_calls:
            print(f"[{pretty_type}] {event.source}")
        return
    payload = event.payload
    tail = ""
    if payload:
        tail = f" {payload}"
    print(f"[{pretty_type}] {event.source}" + tail)


def main() -> None:
    require_api_key()

    from blackgeorge import Desk, Job, Worker, Workforce
    from blackgeorge.collaboration import (
        Blackboard,
        Channel,
        blackboard_write_tool,
        channel_receive_tool,
        channel_send_tool,
    )
    from blackgeorge.workflow import Parallel, Step
    from examples.coding_agent.schema import ChangeReport
    from examples.coding_agent.tools import (
        ask_user,
        list_files,
        modified_files,
        read_file,
        recall,
        remember,
        restore_modified_files,
        search_docs,
        write_file,
    )

    storage_dir = str(Path(__file__).resolve().parent / ".blackgeorge")
    stream_enabled = os.getenv("BLACKGEORGE_STREAM", "1") == "1"

    channel = Channel()
    blackboard = Blackboard()
    manager_blackboard_write = blackboard_write_tool(blackboard, author="Manager")
    reviewer_blackboard_write = blackboard_write_tool(blackboard, author="Reviewer")
    coder_channel_send = channel_send_tool(channel, sender="Coder")
    reviewer_channel_receive = channel_receive_tool(channel, recipient="Reviewer")

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
        tools=[manager_blackboard_write],
        instructions=(
            "You select the best worker for the task. "
            "Always choose Coder unless the task is only a summary. "
            "Post your decision to the blackboard under 'manager_decision' using blackboard_write."
        ),
    )

    coder = Worker(
        name="Coder",
        model=MODEL_NAME,
        tools=[
            list_files,
            read_file,
            write_file,
            ask_user,
            search_docs,
            remember,
            recall,
            coder_channel_send,
        ],
        instructions=(
            "You are a coding agent working inside a small project. "
            "Use list_files and read_file to inspect the project. "
            "Use search_docs to find relevant code by semantic search. "
            "Use remember/recall to save and retrieve notes. "
            "Spec is in spec.txt. "
            "Before deciding behavior for divide by zero or empty averages, "
            "call ask_user with a specific question in the question field. "
            "Use write_file to apply changes. "
            "When done, send a message to Reviewer via channel_send."
        ),
    )

    reviewer = Worker(
        name="Reviewer",
        model=MODEL_NAME,
        tools=[
            search_docs,
            read_file,
            recall,
            reviewer_channel_receive,
            reviewer_blackboard_write,
        ],
        instructions=(
            "You summarize the changes and provide a structured report. "
            "Use search_docs and read_file to verify changes in the codebase. "
            "Use recall to check any notes left by the Coder. "
            "Use changed_files from the job input to avoid guessing. "
            "Post your review summary to the blackboard under 'review_summary' "
            "using blackboard_write. "
            "Only respond with the schema fields."
        ),
    )

    narrator = Worker(
        name="Narrator",
        model=MODEL_NAME,
        tools=[recall],
        instructions=(
            "You produce a concise human-readable summary of the changes. "
            "Use recall to check notes from the team."
        ),
    )

    workforce = Workforce(
        [coder, reviewer],
        mode="managed",
        name="coding_team",
        manager=manager,
        channel=channel,
        blackboard=blackboard,
    )

    blackboard.write("project_name", "calculator", author="system")
    blackboard.write("task_started", True, author="system")

    job = Job(
        input={
            "task": "Fix calculator behavior and update tests.",
            "project": "Use tools to inspect the project files.",
            "requirements": [
                "Confirm divide-by-zero behavior with ask_user.",
                "Confirm empty-average behavior with ask_user.",
                "Apply changes using write_file.",
                "Use search_docs to find relevant code.",
                "Use remember to save important decisions.",
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

        print("\n--- Blackboard State ---")
        bb_keys = blackboard.keys()
        for key in bb_keys:
            print(f"  {key}: {blackboard.read(key)}")

        print("\n--- Channel Messages ---")
        for msg in channel.all_messages():
            print(f"  [{msg.sender} -> {msg.recipient or 'all'}]: {msg.content}")

        flow_job = Job(
            input={
                "source_report": report.content or "",
                "notes": "Generate a summary and a structured change report.",
                "changed_files": modified_files(),
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
                Parallel(
                    Step(reviewer, job_builder=review_job),
                    Step(narrator, job_builder=narrative_job),
                ),
            ]
        )

        flow_report = flow.run(flow_job)

        print("\n--- Blackboard State After Flow ---")
        bb_keys = blackboard.keys()
        for key in bb_keys:
            print(f"  {key}: {blackboard.read(key)}")

        print("\nFinal report status:", flow_report.status)
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
