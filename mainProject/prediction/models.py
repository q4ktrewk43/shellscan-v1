from django.db import models


class ShellfishAdvisory(models.Model):
	bulletin_number = models.CharField(max_length=120)
	bulletin_year = models.PositiveSmallIntegerField()
	title = models.CharField(max_length=500)
	advisory_date = models.DateField(null=True, blank=True)
	status = models.CharField(max_length=120, blank=True)
	affected_areas = models.TextField(blank=True)
	important_information = models.TextField(blank=True)
	source_url = models.URLField(max_length=1000)
	pdf_url = models.URLField(max_length=1000, blank=True)
	pdf_filename = models.CharField(max_length=255, blank=True)
	pdf_content_type = models.CharField(max_length=120, blank=True)
	pdf_size = models.PositiveBigIntegerField(null=True, blank=True)
	pdf_sha256 = models.CharField(max_length=64, blank=True)
	pdf_content = models.BinaryField(null=True, editable=False)
	pdf_text_extracted = models.BooleanField(default=False)
	verification_method = models.CharField(max_length=40, blank=True)
	first_seen_at = models.DateTimeField(auto_now_add=True)
	last_checked_at = models.DateTimeField(auto_now=True)
	pdf_checked_at = models.DateTimeField(null=True, blank=True)

	class Meta:
		ordering = ('-advisory_date', '-first_seen_at')
		constraints = [
			models.UniqueConstraint(
				fields=('bulletin_number', 'bulletin_year'),
				name='unique_shellfish_bulletin_year',
			),
		]

	def __str__(self):
		return self.bulletin_number
