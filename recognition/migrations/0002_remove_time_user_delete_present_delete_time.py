# Generated by Django 4.2.5 on 2023-10-22 06:39

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("recognition", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="time",
            name="user",
        ),
        migrations.DeleteModel(
            name="Present",
        ),
        migrations.DeleteModel(
            name="Time",
        ),
    ]
