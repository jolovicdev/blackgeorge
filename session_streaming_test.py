import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from blackgeorge import Desk, Event, Job, Message, Worker, tool
from blackgeorge.event_bus import EventBus


@tool()
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 22°C."


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def test_reasoning_content_sync():
    print_subsection("Testing reasoning_content (sync)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=":memory:")
    worker = Worker(
        name="test_worker",
        instructions="You are a helpful assistant.",
        tools=[get_weather],
    )

    job = Job(
        input="What is 2+2? Keep it brief.",
        thinking={"type": "enabled", "budget_tokens": 1000},
    )

    try:
        report = desk.run(worker, job)

        print(f"Status: {report.status}")
        print(f"Content: {report.content}")
        if report.reasoning_content:
            print(f"\nReasoning Content (first 500 chars):\n{report.reasoning_content[:500]}...")
        else:
            print("\nNo reasoning_content returned")

        if report.metrics:
            print(f"\nMetrics: {report.metrics}")

    except Exception as e:
        print(f"ERROR: {e}")


async def test_reasoning_content_async():
    print_subsection("Testing reasoning_content (async)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=":memory:")
    worker = Worker(
        name="test_worker",
        instructions="You are a helpful assistant.",
        tools=[get_weather],
    )

    job = Job(
        input="What is 2+2? Keep it brief.",
        thinking={"type": "enabled", "budget_tokens": 1000},
    )

    try:
        report = await desk.arun(worker, job)

        print(f"Status: {report.status}")
        print(f"Content: {report.content}")
        if report.reasoning_content:
            print(f"\nReasoning Content (first 500 chars):\n{report.reasoning_content[:500]}...")
        else:
            print("\nNo reasoning_content returned")

        if report.metrics:
            print(f"\nMetrics: {report.metrics}")

    except Exception as e:
        print(f"ERROR: {e}")


def test_streaming_sync():
    print_subsection("Testing streaming with real-time output (sync)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    event_bus = EventBus()
    collected_tokens = []

    def on_token(event: Event) -> None:
        token = event.payload.get("token", "")
        if token:
            print(token, end="", flush=True)
            collected_tokens.append(token)

    event_bus.subscribe("stream.token", on_token)

    desk = Desk(
        model="deepseek/deepseek-reasoner",
        storage_dir=":memory:",
        event_bus=event_bus,
    )
    worker = Worker(
        name="test_worker",
        instructions="You are a helpful assistant. Keep responses brief.",
    )

    job = Job(input="Count from 1 to 5.")
    print("Response: ", end="", flush=True)

    try:
        report = desk.run(worker, job, stream=True)
        print()  # New line after streaming
        print(f"Status: {report.status}")
        print(f"Number of events: {len(report.events)}")
        print(f"Collected tokens: {''.join(collected_tokens)}")
        print(f"Event types in report: {[e.type for e in report.events[:10]]}")

    except Exception as e:
        print(f"\nERROR: {e}")


async def test_streaming_async():
    print_subsection("Testing streaming with real-time output (async)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    event_bus = EventBus()
    collected_tokens = []

    def on_token(event: Event) -> None:
        token = event.payload.get("token", "")
        if token:
            print(token, end="", flush=True)
            collected_tokens.append(token)

    event_bus.subscribe("stream.token", on_token)

    desk = Desk(
        model="deepseek/deepseek-reasoner",
        storage_dir=":memory:",
        event_bus=event_bus,
    )
    worker = Worker(
        name="test_worker",
        instructions="You are a helpful assistant. Keep responses brief.",
    )

    job = Job(input="Count from 1 to 5.")
    print("Response: ", end="", flush=True)

    try:
        report = await desk.arun(worker, job, stream=True)
        print()  # New line after streaming
        print(f"Status: {report.status}")
        print(f"Number of events: {len(report.events)}")
        print(f"Collected tokens: {''.join(collected_tokens)}")

    except Exception as e:
        print(f"\nERROR: {e}")


def test_initial_messages():
    print_subsection("Testing initial_messages (sync)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=":memory:")
    worker = Worker(
        name="test_worker",
        instructions="You are a helpful assistant. Keep responses brief.",
    )

    initial_messages = [
        Message(role="user", content="My favorite color is blue."),
        Message(role="assistant", content="I'll remember that your favorite color is blue."),
    ]

    job = Job(input="What is my favorite color?", initial_messages=initial_messages)

    try:
        report = desk.run(worker, job)

        print(f"Status: {report.status}")
        print(f"Content: {report.content}")
        print(f"Number of messages in context: {len(report.messages)}")

    except Exception as e:
        print(f"ERROR: {e}")


async def test_initial_messages_async():
    print_subsection("Testing initial_messages (async)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=":memory:")
    worker = Worker(
        name="test_worker",
        instructions="You are a helpful assistant. Keep responses brief.",
    )

    initial_messages = [
        Message(role="user", content="My favorite color is blue."),
        Message(role="assistant", content="I'll remember that your favorite color is blue."),
    ]

    job = Job(input="What is my favorite color?", initial_messages=initial_messages)

    try:
        report = await desk.arun(worker, job)

        print(f"Status: {report.status}")
        print(f"Content: {report.content}")
        print(f"Number of messages in context: {len(report.messages)}")

    except Exception as e:
        print(f"ERROR: {e}")


def test_extra_job_params():
    print_subsection("Testing extra job parameters (sync)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=":memory:")
    worker = Worker(
        name="test_worker",
        instructions="You are a helpful assistant. Keep responses brief.",
    )

    job = Job(
        input="Say hello in one word.",
        drop_params=True,
        extra_body={"custom_param": "value"},
    )

    try:
        report = desk.run(worker, job)

        print(f"Status: {report.status}")
        print(f"Content: {report.content}")

    except Exception as e:
        print(f"ERROR: {e}")


async def test_extra_job_params_async():
    print_subsection("Testing extra job parameters (async)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=":memory:")
    worker = Worker(
        name="test_worker",
        instructions="You are a helpful assistant. Keep responses brief.",
    )

    job = Job(
        input="Say hello in one word.",
        drop_params=True,
        extra_body={"custom_param": "value"},
    )

    try:
        report = await desk.arun(worker, job)

        print(f"Status: {report.status}")
        print(f"Content: {report.content}")

    except Exception as e:
        print(f"ERROR: {e}")


def test_session_sync():
    print_subsection("Testing session (sync)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=tmpdir)
            worker = Worker(
                name="test_worker",
                instructions="You are a helpful assistant. Keep responses brief.",
            )

            session = desk.session(worker, metadata={"test": "session_test"})

            if session is None:
                print("ERROR: Failed to create session")
                return

            print(f"Session ID: {session.session_id}")

            print("\n--- First message ---")
            report1 = session.run("My name is Alice.")
            print(f"Response 1: {report1.content}")

            print("\n--- Second message ---")
            report2 = session.run("What is my name?")
            print(f"Response 2: {report2.content}")

            print("\n--- Session history ---")
            history = session.history()
            print(f"Number of messages in history: {len(history)}")

            session.close()
            print("\nSession closed successfully")

        except Exception as e:
            print(f"ERROR: {e}")


async def test_session_async():
    print_subsection("Testing session (async)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=tmpdir)
            worker = Worker(
                name="test_worker",
                instructions="You are a helpful assistant. Keep responses brief.",
            )

            session = desk.session(worker, metadata={"test": "session_test"})

            if session is None:
                print("ERROR: Failed to create session")
                return

            print(f"Session ID: {session.session_id}")

            print("\n--- First message ---")
            report1 = await session.arun("My name is Bob.")
            print(f"Response 1: {report1.content}")

            print("\n--- Second message ---")
            report2 = await session.arun("What is my name?")
            print(f"Response 2: {report2.content}")

            print("\n--- Session history ---")
            history = session.history()
            print(f"Number of messages in history: {len(history)}")

            session.close()
            print("\nSession closed successfully")

        except Exception as e:
            print(f"ERROR: {e}")


def test_session_stream():
    print_subsection("Testing session with real-time streaming (sync)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            event_bus = EventBus()

            def on_token(event: Event) -> None:
                token = event.payload.get("token", "")
                if token:
                    print(token, end="", flush=True)

            event_bus.subscribe("stream.token", on_token)

            desk = Desk(
                model="deepseek/deepseek-reasoner",
                storage_dir=tmpdir,
                event_bus=event_bus,
            )
            worker = Worker(
                name="test_worker",
                instructions="You are a helpful assistant. Keep responses brief.",
            )

            session = desk.session(worker)

            if session is None:
                print("ERROR: Failed to create session")
                return

            print(f"Session ID: {session.session_id}")

            print("\n--- Streaming interaction ---")
            print("Response: ", end="", flush=True)
            list(session.stream_run("Count from 1 to 3."))
            print()  # New line after streaming

            session.close()
            print("\nSession closed successfully")

        except Exception as e:
            print(f"\nERROR: {e}")


async def test_session_stream_async():
    print_subsection("Testing session with real-time streaming (async)")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            event_bus = EventBus()

            def on_token(event: Event) -> None:
                token = event.payload.get("token", "")
                if token:
                    print(token, end="", flush=True)

            event_bus.subscribe("stream.token", on_token)

            desk = Desk(
                model="deepseek/deepseek-reasoner",
                storage_dir=tmpdir,
                event_bus=event_bus,
            )
            worker = Worker(
                name="test_worker",
                instructions="You are a helpful assistant. Keep responses brief.",
            )

            session = desk.session(worker)

            if session is None:
                print("ERROR: Failed to create session")
                return

            print(f"Session ID: {session.session_id}")

            print("\n--- Streaming interaction ---")
            print("Response: ", end="", flush=True)
            events = []
            async for event in session.astream_run("Count from 1 to 3."):
                events.append(event)
            print()  # New line after streaming

            session.close()
            print("\nSession closed successfully")

        except Exception as e:
            print(f"\nERROR: {e}")


def test_thinking_parameter():
    print_subsection("Testing thinking parameter variations")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIPPED: DEEPSEEK_API_KEY not set")
        return

    desk = Desk(model="deepseek/deepseek-reasoner", storage_dir=":memory:")
    worker = Worker(name="test_worker", instructions="You are a helpful assistant.")

    print("\nTest 1: thinking with budget_tokens")
    job1 = Job(input="2+2?", thinking={"type": "enabled", "budget_tokens": 500})
    try:
        report1 = desk.run(worker, job1)
        print(f"Content: {report1.content}")
        print(f"Has reasoning: {bool(report1.reasoning_content)}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\nTest 2: thinking without budget")
    job2 = Job(input="2+2?", thinking={"type": "enabled"})
    try:
        report2 = desk.run(worker, job2)
        print(f"Content: {report2.content}")
        print(f"Has reasoning: {bool(report2.reasoning_content)}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\nTest 3: no thinking parameter")
    job3 = Job(input="2+2?")
    try:
        report3 = desk.run(worker, job3)
        print(f"Content: {report3.content}")
        print(f"Has reasoning: {bool(report3.reasoning_content)}")
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    print("\n" + "#" * 60)
    print("#  BLACKGEORGE-AGENTS NEW FEATURES TEST SUITE")
    print("#" * 60)

    has_deepseek = bool(os.environ.get("DEEPSEEK_API_KEY"))

    print("\nAPI Key Status:")
    print(f"  DEEPSEEK_API_KEY:    {'SET' if has_deepseek else 'NOT SET'}")

    if not has_deepseek:
        print("\nERROR: DEEPSEEK_API_KEY must be set!")
        print("\nTo run tests, set API key:")
        print("  export DEEPSEEK_API_KEY='your-key'")
        return

    print_section("SYNCHRONOUS TESTS")
    test_reasoning_content_sync()
    test_streaming_sync()
    test_initial_messages()
    test_extra_job_params()
    test_thinking_parameter()
    test_session_sync()
    test_session_stream()

    print_section("ASYNCHRONOUS TESTS")

    async def run_all_async():
        await test_reasoning_content_async()
        await test_streaming_async()
        await test_initial_messages_async()
        await test_extra_job_params_async()
        await test_session_async()
        await test_session_stream_async()

    asyncio.run(run_all_async())

    print("\n" + "#" * 60)
    print("#  ALL TESTS COMPLETED")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
