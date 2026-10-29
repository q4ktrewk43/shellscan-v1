import logging
import threading
import time
from datetime import timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from django.db import close_old_connections
from django.utils import timezone

from prediction.models import ShellfishAdvisory
from prediction.services.bfar import SYNC_INTERVAL, sync_latest_advisory

logger = logging.getLogger(__name__)
scheduler = None
SYNC_RETRY_INTERVAL = timedelta(minutes=15)
_on_demand_sync_lock = threading.Lock()
_last_on_demand_sync_at = None


def _run_sync(force=False):
    close_old_connections()
    try:
        return sync_latest_advisory(force=force)
    finally:
        close_old_connections()


def schedule_sync(force=False):
    global _last_on_demand_sync_at

    latest = ShellfishAdvisory.objects.defer('pdf_content').first()
    now = timezone.now()
    stale = latest is None or now - latest.last_checked_at >= SYNC_INTERVAL
    if (
        not stale
        and not force
    ):
        return {
            'attempted': False,
            'changed': False,
            'cached': True,
            'queued': False,
            'stale': False,
        }

    if scheduler is None or not scheduler.running:
        raise RuntimeError('BFAR advisory scheduler is not running.')

    with _on_demand_sync_lock:
        now_monotonic = time.monotonic()
        if (
            not force
            and _last_on_demand_sync_at is not None
            and now_monotonic - _last_on_demand_sync_at < SYNC_RETRY_INTERVAL.total_seconds()
        ):
            retry_after = int(
                SYNC_RETRY_INTERVAL.total_seconds()
                - (now_monotonic - _last_on_demand_sync_at)
            )
            return {
                'attempted': False,
                'changed': False,
                'cached': bool(latest),
                'queued': False,
                'running': True,
                'stale': stale,
                'retry_after_seconds': max(retry_after, 1),
            }

        scheduler.add_job(
            _run_sync,
            'date',
            run_date=now,
            id='sync_bfar_advisory_on_demand',
            name='Sync BFAR Shellfish Advisory (on demand)',
            kwargs={'force': force},
            replace_existing=True,
        )
        _last_on_demand_sync_at = now_monotonic

    logger.info('Queued BFAR advisory sync (force=%s)', force)
    return {
        'attempted': False,
        'changed': False,
        'cached': bool(latest),
        'queued': True,
        'running': True,
        'stale': stale,
    }


def start_scheduler():
    global scheduler
    if scheduler is not None and scheduler.running:
        return

    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            _run_sync,
            'interval',
            hours=6,
            id='sync_bfar_advisory',
            name='Sync BFAR Shellfish Advisory',
            replace_existing=True,
            misfire_grace_time=60,
            coalesce=True,
        )
        scheduler.start()
        logger.info('BFAR advisory scheduler started: syncs every 6 hours')
    except Exception as exc:
        logger.error(f'Failed to start BFAR scheduler: {exc}')


def stop_scheduler():
    global scheduler
    if scheduler is not None and scheduler.running:
        scheduler.shutdown()
        logger.info('BFAR advisory scheduler stopped')
