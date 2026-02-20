import asyncio
from pathlib import Path
from annolid.core.agent.bus.events import InboundMessage
from annolid.core.agent.bus.queue import MessageBus
from annolid.core.agent.cron.types import CronJob, CronPayload, CronSchedule
from annolid.core.agent.cron.service import CronService


async def main():
    bus = MessageBus()
    store_path = Path("test_jobs.json")
    if store_path.exists():
        store_path.unlink()

    async def _on_cron_job(job: CronJob) -> str:
        channel = str(job.payload.channel or "cli")
        chat_id = str(job.payload.to or "direct")
        msg = str(job.payload.message or "").strip()
        await bus.publish_inbound(
            InboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=msg,
            )
        )
        return "Inbound generated"

    service = CronService(store_path=store_path, on_job=_on_cron_job)
    await service.start()

    # Schedule a job
    service.add_job(
        name="test_job",
        schedule=CronSchedule(kind="at", at_ms=0),  # immediate
        payload=CronPayload(message="Fetch news"),
    )

    # Wait for the service to trigger the job
    await asyncio.sleep(1.5)

    try:
        inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert inbound.content == "Fetch news"
        print("PASS: Cron job successfully published to AgentBus")
    except asyncio.TimeoutError:
        print("FAIL: No message on bus")
    finally:
        await service.stop()
        if store_path.exists():
            store_path.unlink()


asyncio.run(main())
