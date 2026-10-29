import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from django.db import migrations


def seed_bulletin_history(apps, schema_editor):
    ShellfishAdvisory = apps.get_model('prediction', 'ShellfishAdvisory')
    database = schema_editor.connection.alias
    data_path = (
        Path(__file__).resolve().parents[1]
        / 'data'
        / 'cached_advisory_history.json'
    )
    with data_path.open(encoding='utf-8') as history_file:
        history = json.load(history_file)

    for record in history:
        record = dict(record)
        bulletin_number = record.pop('bulletin_number')
        bulletin_year = record.pop('bulletin_year')
        if record['advisory_date']:
            record['advisory_date'] = date.fromisoformat(record['advisory_date'])
        ShellfishAdvisory.objects.using(database).get_or_create(
            bulletin_number=bulletin_number,
            bulletin_year=bulletin_year,
            defaults=record,
        )

    pdf_path = (
        Path(__file__).resolve().parents[1]
        / 'data'
        / 'SB-no.-25-dated-16-Oct-2026.pdf'
    )
    pdf_content = pdf_path.read_bytes()
    if not pdf_content.startswith(b'%PDF'):
        raise RuntimeError(f'Cached advisory PDF is invalid: {pdf_path}')

    pdf_url = (
        'https://www.bfar.da.gov.ph/wp-content/uploads/2026/10/'
        'SB-no.-25_dated-16-Oct-2026.pdf'
    )
    latest, _ = ShellfishAdvisory.objects.using(database).update_or_create(
        bulletin_number='SB 25',
        bulletin_year=2026,
        defaults={
            'title': 'Shellfish Bulletin No. 25 Series of 2026',
            'advisory_date': date(2026, 10, 16),
            'status': 'PSP positive above the regulatory limit',
            'affected_areas': (
                'Dumanquillas Bay, Zamboanga del Sur\n'
                'Tantanang Bay, Zamboanga Sibugay Province\n'
                'Matarinao Bay, Eastern Samar'
            ),
            'important_information': (
                'BFAR advises: Do not sell, do not buy, and do not eat '
                'shellfish from affected areas.'
            ),
            'source_url': (
                'https://www.bfar.da.gov.ph/shellfish-bulletin-no-02-2022/'
            ),
            'pdf_url': pdf_url,
            'pdf_filename': 'SB-no.-25_dated-16-Oct-2026.pdf',
            'pdf_content_type': 'application/pdf',
            'pdf_size': len(pdf_content),
            'pdf_sha256': hashlib.sha256(pdf_content).hexdigest().upper(),
            'pdf_content': pdf_content,
            'pdf_text_extracted': False,
            'verification_method': 'official_bfar_page',
        },
    )
    source_checked_at = datetime(2026, 10, 19, 2, 37, 2, tzinfo=timezone.utc)
    ShellfishAdvisory.objects.using(database).filter(pk=latest.pk).update(
        last_checked_at=source_checked_at,
        pdf_checked_at=source_checked_at,
    )


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0006_cache_seeded_bulletin_pdf'),
    ]

    operations = [
        migrations.RunPython(seed_bulletin_history, migrations.RunPython.noop),
    ]
