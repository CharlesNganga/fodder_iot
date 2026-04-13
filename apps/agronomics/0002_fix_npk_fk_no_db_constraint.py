from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):
    dependencies = [
        ('agronomics', '0001_initial'),
        ('telemetry', '0001_initial'),
    ]
    operations = [
        migrations.AlterField(
            model_name='npkprediction',
            name='telemetry_reading',
            field=models.OneToOneField(
                on_delete=django.db.models.deletion.CASCADE,
                related_name='npk_prediction',
                to='telemetry.dailyiottelemetry',
                db_constraint=False,
                help_text='TimescaleDB does not support FK constraints on hypertables.',
            ),
        ),
    ]