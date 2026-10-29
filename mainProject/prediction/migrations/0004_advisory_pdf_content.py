from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0003_bulletin_verification'),
    ]

    operations = [
        migrations.AddField(
            model_name='shellfishadvisory',
            name='pdf_content',
            field=models.BinaryField(editable=False, null=True),
        ),
    ]
