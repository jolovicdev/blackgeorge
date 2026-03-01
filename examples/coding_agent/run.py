import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coding agent example")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Non-interactive prompt to run (skips interactive mode)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    parser.add_argument(
        "--swarm",
        action="store_true",
        help="Use swarm mode instead of managed mode",
    )
    return parser.parse_args()


def require_api_key() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")


def print_event(event) -> None:
    from blackgeorge import EventType

    if event.type == EventType.STREAM_TOKEN:
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
    if event.type == EventType.TOOL_STARTED:
        tool_call_id = event.payload.get("tool_call_id")
        if tool_call_id:
            _tool_state["active"].add(tool_call_id)
            _tool_state["started_at"][tool_call_id] = time.perf_counter()
            if len(_tool_state["active"]) == 2:
                _tool_state["parallel_batches"] += 1
                active_list = ", ".join(sorted(_tool_state["active"]))
                print(f"[PARALLEL] tools in flight: {active_list}")
    if event.type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}:
        tool_call_id = event.payload.get("tool_call_id")
        cancelled = event.payload.get("cancelled", False)
        if tool_call_id:
            started = _tool_state["started_at"].pop(tool_call_id, None)
            _tool_state["active"].discard(tool_call_id)
            if started is not None:
                elapsed = time.perf_counter() - started
                status = "CANCELLED" if cancelled else "DONE"
                print(f"[TOOL TIMING] {tool_call_id} {elapsed:.3f}s ({status})")
    if event.type == EventType.ASSISTANT_MESSAGE:
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
    if event.type == EventType.WORKER_PAUSED:
        pending_type = event.payload.get("pending_action_type", "unknown")
        print(f"[{pretty_type}] {event.source}: pending_action={pending_type}")
        return
    if event.type == EventType.WORKER_CONTEXT_SUMMARIZED:
        summarized = event.payload.get("summarized_messages", 0)
        kept = event.payload.get("kept_messages", 0)
        print(f"[{pretty_type}] {event.source}: summarized={summarized}, kept={kept}")
        return
    payload = event.payload
    tail = ""
    if payload:
        tail = f" {payload}"
    print(f"[{pretty_type}] {event.source}" + tail)


def main() -> None:
    args = parse_args()
    require_api_key()

    from blackgeorge import (
        Desk,
        EventType,
        Job,
        ToolExecutionError,
        Worker,
        Workforce,
    )
    from blackgeorge.collaboration import (
        Blackboard,
        Channel,
        blackboard_write_tool,
        channel_receive_tool,
        channel_send_tool,
    )
    from blackgeorge.tools import transfer_to_agent_tool
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
    stream_enabled = not args.no_stream and os.getenv("BLACKGEORGE_STREAM", "1") == "1"
    use_swarm = args.swarm
    non_interactive_prompt = args.prompt

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
        max_context_messages=15,
    )
    keep_changes = os.getenv("PRESERVE_EXAMPLE_CHANGES") == "1"

    event_types = {
        EventType.RUN_STARTED,
        EventType.RUN_PAUSED,
        EventType.RUN_RESUMED,
        EventType.RUN_COMPLETED,
        EventType.RUN_FAILED,
        EventType.WORKFORCE_STARTED,
        EventType.WORKFORCE_COMPLETED,
        EventType.WORKER_STARTED,
        EventType.WORKER_COMPLETED,
        EventType.WORKER_FAILED,
        EventType.WORKER_PAUSED,
        EventType.WORKER_CONTEXT_SUMMARIZED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.TOOL_FAILED,
        EventType.TOOL_CONFIRMATION_REQUESTED,
        EventType.TOOL_USER_INPUT_REQUESTED,
        EventType.ASSISTANT_MESSAGE,
        EventType.STREAM_TOKEN,
    }

    def on_event(event) -> None:
        if event.type in event_types:
            print_event(event)

    desk.event_bus.subscribe("*", on_event)

    handoff_tool = transfer_to_agent_tool(["Coder", "Reviewer"])

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
            handoff_tool,
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
            "When done, send a message to Reviewer via channel_send. "
            "In swarm mode, use transfer_to_agent to hand off to Reviewer."
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
            handoff_tool,
        ],
        instructions=(
            "You summarize the changes and provide a structured report. "
            "Use search_docs and read_file to verify changes in the codebase. "
            "Use recall to check any notes left by the Coder. "
            "Use changed_files from the job input to avoid guessing. "
            "Post your review summary to the blackboard under 'review_summary' "
            "using blackboard_write. "
            "Only respond with the schema fields. "
            "In swarm mode, use transfer_to_agent to hand off back to Coder if needed."
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

    if use_swarm:
        workforce = Workforce(
            [coder, reviewer],
            mode="swarm",
            name="coding_swarm",
            channel=channel,
            blackboard=blackboard,
        )
    else:
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

    task_text = non_interactive_prompt or "Fix calculator behavior and update tests."

    job = Job(
        input={
            "task": task_text,
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

    auto_confirm = non_interactive_prompt is not None

    try:
        report = desk.run(workforce, job, stream=stream_enabled)

        while report.status == "paused" and report.pending_action is not None:
            action = report.pending_action
            if action.type == "handoff":
                print(f"[HANDOFF] -> {action.prompt}")
                report = desk.resume(report, "", stream=stream_enabled)
            elif action.type == "confirmation":
                if auto_confirm:
                    print(f"[AUTO-CONFIRM] {action.prompt} -> y")
                    decision = True
                else:
                    decision = input(f"{action.prompt} [y/n]: ").strip().lower() in {"y", "yes"}
                report = desk.resume(report, decision, stream=stream_enabled)
            else:
                if auto_confirm:
                    default_response = "proceed with safe defaults"
                    print(f"[AUTO-INPUT] {action.prompt} -> {default_response}")
                    decision = default_response
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
    except ToolExecutionError as e:
        print(f"Tool execution failed: {e.tool_name}: {e}")
    finally:
        desk.event_bus.unsubscribe("*", on_event)
        if not keep_changes:
            changed = modified_files()
            if changed:
                restored = restore_modified_files()
                print("Reverted example files:", ", ".join(restored))


if __name__ == "__main__":
    main()
