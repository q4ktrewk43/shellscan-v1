from datetime import date, datetime, timezone

from django.db import migrations


def seed_cached_advisory(apps, schema_editor):
    ShellfishAdvisory = apps.get_model('prediction', 'ShellfishAdvisory')
    if ShellfishAdvisory.objects.exists():
        return

    advisory = ShellfishAdvisory.objects.create(
        bulletin_number='SB 22',
        bulletin_year=2026,
        title='Shellfish Bulletin No. 22 - 16 September 2026',
        advisory_date=date(2026, 9, 16),
        status='PSP positive above the regulatory limit',
        affected_areas=(
            'Dumanquillas Bay, Zamboanga del Sur\n'
            'Tantanang Bay, Zamboanga Sibugay Province\n'
            'Matarinao Bay, Eastern Samar\n'
            'Zumarraga Island, Samar'
        ),
        important_information=(
            'Shellfish samples from the listed areas tested positive for '
            'paralytic shellfish poison (PSP). This is a dated cached '
            'bulletin; verify current conditions with BFAR.'
        ),
        source_url=(
            'https://www.bfar.da.gov.ph/wp-content/uploads/2026/09/'
            'SB-NO.-22-dated-16-Sept-2026.pdf'
        ),
        pdf_url=(
            'https://www.bfar.da.gov.ph/wp-content/uploads/2026/09/'
            'SB-NO.-22-dated-16-Sept-2026.pdf'
        ),
        pdf_filename='SB-NO.-22-dated-16-Sept-2026.pdf',
        pdf_content_type='application/pdf',
        pdf_size=814626,
        verification_method='cached_snapshot',
    )
    cached_at = datetime(2026, 9, 26, 9, 40, 21, 674319, tzinfo=timezone.utc)
    ShellfishAdvisory.objects.filter(pk=advisory.pk).update(
        first_seen_at=cached_at,
        last_checked_at=cached_at,
    )


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0004_advisory_pdf_content'),
    ]

    operations = [
        migrations.RunPython(seed_cached_advisory, migrations.RunPython.noop),
    ]
