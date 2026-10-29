from pathlib import Path

from django.db import migrations


def cache_seeded_bulletin_pdf(apps, schema_editor):
    ShellfishAdvisory = apps.get_model('prediction', 'ShellfishAdvisory')
    advisory = ShellfishAdvisory.objects.filter(
        bulletin_number='SB 22',
        bulletin_year=2026,
        pdf_content__isnull=True,
    ).first()
    if advisory is None:
        return

    pdf_path = (
        Path(__file__).resolve().parents[1]
        / 'data'
        / 'SB-NO.-22-dated-16-Sept-2026.pdf'
    )
    pdf_content = pdf_path.read_bytes()
    if not pdf_content.startswith(b'%PDF'):
        raise RuntimeError(f'Cached advisory PDF is invalid: {pdf_path}')

    ShellfishAdvisory.objects.filter(pk=advisory.pk).update(
        pdf_content=pdf_content,
        pdf_size=len(pdf_content),
        pdf_sha256='8326307FD75CC74162D58BF85B8BE374B9E8DEE96F841195D80727F9370551EA',
    )


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0005_seed_cached_advisory'),
    ]

    operations = [
        migrations.RunPython(cache_seeded_bulletin_pdf, migrations.RunPython.noop),
    ]
