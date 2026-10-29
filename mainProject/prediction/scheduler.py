import logging
from apscheduler.schedulers.background import BackgroundScheduler
from django.db import close_old_connections
from django.utils import timezone

from prediction.models import ShellfishAdvisory
from prediction.services.bfar import SYNC_INTERVAL, sync_latest_advisory

logger = logging.getLogger(__name__)
scheduler = None


def _run_sync(force=False):
    close_old_connections()
    try:
        return sync_latest_advisory(force=force)
    finally:
        close_old_connections()


def schedule_sync(force=False):
    latest = ShellfishAdvisory.objects.defer('pdf_content').first()
    if (
        latest
        and not force
        and timezone.now() - latest.last_checked_at < SYNC_INTERVAL
    ):
        return {
            'attempted': False,
            'changed': False,
            'cached': True,
            'queued': False,
        }

    if scheduler is None or not scheduler.running:
        raise RuntimeError('BFAR advisory scheduler is not running.')

    scheduler.add_job(
        _run_sync,
        'date',
        run_date=timezone.now(),
        id='sync_bfar_advisory_on_demand',
        name='Sync BFAR Shellfish Advisory (on demand)',
        kwargs={'force': force},
        replace_existing=True,
    )
    logger.info('Queued BFAR advisory sync (force=%s)', force)
    return {
        'attempted': False,
        'changed': False,
        'cached': bool(latest),
        'queued': True,
        'running': True,
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
