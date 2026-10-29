import importlib
import io
import os
from datetime import date, timedelta
from unittest.mock import Mock, patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from PIL import Image
from requests.exceptions import Timeout

from prediction.views import _ensure_probabilities, _preprocess_image
from prediction.models import ShellfishAdvisory


class PredictViewTests(TestCase):
    def test_preprocess_image_preserves_efficientnet_v2_input_values(self):
        image = Image.new('RGB', (224, 224), color=(0, 128, 255))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        processed = _preprocess_image(buffer)

        self.assertEqual(processed.shape, (1, 224, 224, 3))
        np.testing.assert_array_equal(processed[0, 0, 0], [0.0, 128.0, 255.0])

    def test_softmax_normalizes_logits_before_confidence_is_reported(self):
        logits = np.array([3.0, 1.0, -1.0, -2.0], dtype=np.float32)
        probabilities = _ensure_probabilities(logits)

        self.assertEqual(probabilities.shape, logits.shape)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        self.assertGreater(float(probabilities[0]), 0.5)
    
    def test_ensure_probabilities_handles_already_softmax_output(self):
        """Test that already-normalized output (sum=1.0) is not double-normalized."""
        # Model output that already sums to 1.0 (from softmax layer)
        probabilities_input = np.array([0.1, 0.41, 0.44, 0.05], dtype=np.float32)
        probabilities = _ensure_probabilities(probabilities_input)
        
        # Should remain approximately the same (not double-softmaxed)
        self.assertAlmostEqual(float(probabilities[2]), 0.44, places=3)  # max should stay ~0.44
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_predict_endpoint_accepts_image_and_returns_prediction(self):
        image = Image.new('RGB', (224, 224), color=(25, 35, 45))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        uploaded_image = SimpleUploadedFile(
            'shell_test.png',
            buffer.read(),
            content_type='image/png',
        )

        response = self.client.post(
            reverse('predict_shell'),
            {'image': uploaded_image},
            format='multipart',
        )

        print('DEBUG_STATUS', response.status_code)
        print('DEBUG_BODY', response.content.decode())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn('prediction', payload)
        self.assertIn('confidence', payload)
        self.assertIn('label', payload)
        self.assertIn('probabilities', payload)
        self.assertIsInstance(payload['probabilities'], dict)
        self.assertEqual(len(payload['probabilities']), 8)
        self.assertGreaterEqual(float(payload['confidence']), 0.0)
        self.assertLessEqual(float(payload['confidence']), 1.0)
        self.assertNotEqual(payload['prediction'], 'conus-textile')


class HostConfigurationTests(TestCase):
    def test_production_hosts_allow_railway_domains_by_default(self):
        import mainProject.settings as settings_module

        original_allowed_hosts = os.environ.get('DJANGO_ALLOWED_HOSTS')
        original_debug = os.environ.get('DJANGO_DEBUG')
        try:
            os.environ.pop('DJANGO_ALLOWED_HOSTS', None)
            os.environ['DJANGO_DEBUG'] = 'False'
            importlib.reload(settings_module)
            hosts = set(settings_module.ALLOWED_HOSTS)
            self.assertTrue('*' in hosts or '*.up.railway.app' in hosts or '*.railway.app' in hosts)
        finally:
            if original_allowed_hosts is None:
                os.environ.pop('DJANGO_ALLOWED_HOSTS', None)
            else:
                os.environ['DJANGO_ALLOWED_HOSTS'] = original_allowed_hosts
            if original_debug is None:
                os.environ.pop('DJANGO_DEBUG', None)
            else:
                os.environ['DJANGO_DEBUG'] = original_debug
            importlib.reload(settings_module)


class RootRouteTests(TestCase):
    def test_root_and_favicon_routes_do_not_404(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/favicon.ico').status_code, 200)


class ShellfishAdvisoryTests(TestCase):
    def setUp(self):
        ShellfishAdvisory.objects.all().delete()

    @patch('requests.get')
    def test_pdf_endpoint_redirects_directly_to_official_pdf(self, request_get):
        ShellfishAdvisory.objects.create(
            bulletin_number='Bulletin 1',
            bulletin_year=2024,
            title='Shellfish Advisory',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://www.bfar.da.gov.ph/advisory.pdf',
            pdf_filename='advisory.pdf',
        )
        request_get.return_value = type('Response', (), {
            'content': b'%PDF-1.4 test',
            'raise_for_status': lambda self: None,
        })()

        response = self.client.get(reverse('shellfish_advisory_pdf'))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response['Location'],
            'https://www.bfar.da.gov.ph/advisory.pdf',
        )
        self.assertEqual(response['Cache-Control'], 'no-store')
        request_get.assert_not_called()

    @patch('requests.get')
    def test_pdf_endpoint_serves_cached_pdf_without_contacting_bfar(self, request_get):
        ShellfishAdvisory.objects.create(
            bulletin_number='Bulletin 1',
            bulletin_year=2024,
            title='Shellfish Advisory',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://www.bfar.da.gov.ph/advisory.pdf',
            pdf_filename='advisory.pdf',
            pdf_content=b'%PDF-1.4 cached',
        )

        response = self.client.get(reverse('shellfish_advisory_pdf'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'%PDF-1.4 cached')
        self.assertEqual(response['Content-Disposition'], 'inline; filename="advisory.pdf"')
        self.assertEqual(response['Cache-Control'], 'private, max-age=300')
        self.assertNotIn('X-Frame-Options', response)
        self.assertIn('frame-ancestors', response['Content-Security-Policy'])
        request_get.assert_not_called()

    @patch('requests.get')
    def test_pdf_endpoint_rejects_non_official_pdf_url(self, request_get):
        ShellfishAdvisory.objects.create(
            bulletin_number='Bulletin 1',
            bulletin_year=2024,
            title='Shellfish Advisory',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://example.com/advisory.pdf',
        )

        response = self.client.get(reverse('shellfish_advisory_pdf'))

        self.assertEqual(response.status_code, 502)
        self.assertIn('not an official BFAR URL', response.json()['error'])
        request_get.assert_not_called()

    def test_parse_date_handles_abbreviated_month_names(self):
        from prediction.services.bfar import _parse_date

        self.assertEqual(_parse_date('16 Sept 2026'), date(2026, 9, 16))
        self.assertEqual(_parse_date('16-Sept-2026'), date(2026, 9, 16))

    @patch('prediction.services.bfar.firebase_bulletins.upsert')
    @patch('prediction.services.bfar.fetch_pdf')
    @patch('prediction.services.bfar._archive_candidates')
    @patch('prediction.services.bfar.requests.get')
    def test_sync_detects_bulletin_and_caches_pdf_metadata(
        self, request_get, archive_candidates, fetch_pdf_mock, upsert_mock,
    ):
        listing = type('Response', (), {
            'text': '<html>Shellfish Bulletin</html>',
            'content': b'html',
            'status_code': 200,
            'headers': {},
            'raise_for_status': lambda self: None,
        })()
        request_get.return_value = listing

        archive_candidates.return_value = [{
            'title': 'Shellfish Bulletin No. 12',
            'source_url': 'https://www.bfar.da.gov.ph/bulletin.html',
            'pdf_url': 'https://www.bfar.da.gov.ph/files/bulletin12.pdf',
            'pdf_filename': 'bulletin12.pdf',
            'pdf_content_type': 'application/pdf',
            'pdf_size': 128,
            'pdf_sha256': 'abc123',
            'status': 'OPEN',
            'affected_areas': ['Area A'],
            'important_information': 'Keep cautious.',
        }]
        fetch_pdf_mock.return_value = (
            type('Response', (), {
                'headers': {'Content-Type': 'application/pdf'},
                'content': b'%PDF-1.4 cached',
            })(),
            {
                'status': 'OPEN',
                'affected_areas': ['Area A'],
                'important_information': 'Keep cautious.',
                'bulletin_number': '12',
                'date': date(2026, 8, 24),
                'pdf_text_extracted': True,
            },
        )

        from prediction.services.bfar import sync_latest_advisory
        result = sync_latest_advisory(force=True)

        self.assertTrue(result['changed'])
        advisory = ShellfishAdvisory.objects.get()
        self.assertEqual(advisory.bulletin_number, '12')
        self.assertTrue(advisory.pdf_url.endswith('bulletin12.pdf'))
        self.assertEqual(advisory.pdf_size, 128)
        upsert_mock.assert_called_once()

    def test_sync_reports_running_when_another_sync_is_in_progress(self):
        from prediction.services import bfar as bfar_service

        self.assertTrue(bfar_service._BFAR_SYNC_LOCK.acquire(blocking=False))
        try:
            result = bfar_service.sync_latest_advisory(force=True)
            self.assertTrue(result['running'])
            self.assertIn('already running', result['error'].lower())
        finally:
            bfar_service._BFAR_SYNC_LOCK.release()

    @patch('prediction.services.bfar.firebase_bulletins.upsert')
    @patch('prediction.services.bfar.fetch_pdf')
    @patch('prediction.services.bfar._archive_candidates')
    @patch('prediction.services.bfar.requests.get')
    def test_sync_uses_new_direct_homepage_bulletin_without_archive_crawl(
        self, request_get, archive_candidates, fetch_pdf_mock, upsert_mock,
    ):
        listing = type('Response', (), {
            'text': (
                '<a href="/files/SB-20-22-August-2026.pdf">'
                'Shellfish Bulletin No. 20 - August 22, 2026</a>'
            ),
            'content': b'html',
            'status_code': 200,
            'headers': {},
            'raise_for_status': lambda self: None,
        })()
        request_get.return_value = listing
        def fetch_latest(advisory):
            advisory.update({
                'pdf_filename': 'SB-20-22-August-2026.pdf',
                'pdf_content_type': 'application/pdf',
                'pdf_size': 128,
                'pdf_sha256': 'def456',
            })
            return (
                type('Response', (), {
                    'headers': {'Content-Type': 'application/pdf'},
                    'content': b'%PDF-1.4 cached',
                })(),
                {
                    'status': 'OPEN',
                    'affected_areas': ['Area B'],
                    'important_information': 'Keep cautious.',
                    'bulletin_number': 'SB 20',
                    'date': date(2026, 8, 22),
                    'pdf_text_extracted': True,
                },
            )

        fetch_pdf_mock.side_effect = fetch_latest

        from prediction.services.bfar import sync_latest_advisory
        result = sync_latest_advisory(force=True)

        self.assertTrue(result['changed'])
        self.assertEqual(ShellfishAdvisory.objects.get().bulletin_number, 'SB 20')
        archive_candidates.assert_not_called()
        upsert_mock.assert_called_once()

    @patch('prediction.views.schedule_sync', return_value={
        'attempted': False,
        'changed': False,
        'cached': True,
        'queued': True,
        'running': True,
    })
    @patch('prediction.services.bfar.requests.get')
    def test_advisory_endpoint_returns_cached_record_without_waiting_for_bfar(
        self, request_get, schedule_sync_mock,
    ):
        advisory = ShellfishAdvisory.objects.create(
            bulletin_number='Bulletin 1',
            bulletin_year=2024,
            title='Shellfish Advisory',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://www.bfar.da.gov.ph/advisory.pdf',
        )
        response = self.client.get(reverse('shellfish_advisory'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['advisory']['bulletin_number'], advisory.bulletin_number)
        self.assertTrue(response.json()['sync']['queued'])
        schedule_sync_mock.assert_called_once_with(force=False)
        request_get.assert_not_called()

    @patch('prediction.views.schedule_sync', return_value={
        'attempted': False,
        'changed': False,
        'cached': True,
        'queued': True,
        'running': True,
    })
    @patch('prediction.services.bfar.requests.get')
    def test_advisory_post_queues_forced_refresh_without_waiting_for_bfar(
        self, request_get, schedule_sync_mock,
    ):
        advisory = ShellfishAdvisory.objects.create(
            bulletin_number='Bulletin 1',
            bulletin_year=2024,
            title='Shellfish Advisory',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://www.bfar.da.gov.ph/advisory.pdf',
        )

        response = self.client.post(reverse('shellfish_advisory'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['advisory']['bulletin_number'], advisory.bulletin_number)
        self.assertTrue(response.json()['sync']['queued'])
        schedule_sync_mock.assert_called_once_with(force=True)
        request_get.assert_not_called()

    @patch('prediction.views.schedule_sync', return_value={
        'attempted': False,
        'changed': False,
        'cached': False,
        'queued': True,
        'running': True,
    })
    def test_advisory_endpoint_reports_unavailable_while_initial_sync_is_queued(
        self, _schedule_sync,
    ):
        response = self.client.get(reverse('shellfish_advisory'))

        self.assertEqual(response.status_code, 202)
        self.assertIsNone(response.json()['advisory'])
        self.assertTrue(response.json()['sync']['queued'])

    @patch('prediction.views.schedule_sync', side_effect=RuntimeError('scheduler unavailable'))
    def test_advisory_endpoint_serves_cache_if_background_scheduler_is_unavailable(
        self, _schedule_sync,
    ):
        advisory = ShellfishAdvisory.objects.create(
            bulletin_number='Bulletin 1',
            bulletin_year=2024,
            title='Shellfish Advisory',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://www.bfar.da.gov.ph/advisory.pdf',
        )

        response = self.client.get(reverse('shellfish_advisory'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['advisory']['bulletin_number'], advisory.bulletin_number)
        self.assertIn('scheduler unavailable', response.json()['sync']['error'])

    def test_stale_advisory_response_includes_warning(self):
        ShellfishAdvisory.objects.create(
            bulletin_number='SB 22',
            bulletin_year=2026,
            title='Shellfish Bulletin No. 22',
            advisory_date=date(2026, 9, 16),
            status='PSP positive above the regulatory limit',
            affected_areas='Dumanquillas Bay',
            important_information='Verify current conditions with BFAR.',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://www.bfar.da.gov.ph/advisory.pdf',
        )
        ShellfishAdvisory.objects.update(
            last_checked_at=timezone.now() - timedelta(days=1),
        )

        with patch('prediction.views.schedule_sync', return_value={
            'attempted': False,
            'changed': False,
            'cached': True,
            'queued': False,
            'stale': True,
        }):
            response = self.client.get(reverse('shellfish_advisory'))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['advisory']['bulletin_number'], 'SB 22')
        self.assertTrue(payload['sync']['stale'])
        self.assertIn('may be outdated', payload['sync']['warning'])
        self.assertTrue(payload['advisory']['stale'])
        self.assertIn('may be outdated', payload['advisory']['warning'])

    def test_stale_sync_requests_are_throttled(self):
        from prediction import scheduler as advisory_scheduler

        advisory = ShellfishAdvisory.objects.create(
            bulletin_number='SB 22',
            bulletin_year=2026,
            title='Shellfish Bulletin No. 22',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://www.bfar.da.gov.ph/advisory.pdf',
        )
        ShellfishAdvisory.objects.filter(pk=advisory.pk).update(
            last_checked_at=timezone.now() - timedelta(days=1),
        )
        fake_scheduler = type('Scheduler', (), {
            'running': True,
            'add_job': Mock(),
        })()

        with (
            patch.object(advisory_scheduler, 'scheduler', fake_scheduler),
            patch.object(advisory_scheduler, '_last_on_demand_sync_at', None),
        ):
            first = advisory_scheduler.schedule_sync()
            second = advisory_scheduler.schedule_sync()

        self.assertTrue(first['queued'])
        self.assertFalse(second['queued'])
        self.assertGreater(second['retry_after_seconds'], 0)
        fake_scheduler.add_job.assert_called_once()

    @patch('prediction.services.bfar.requests.get', side_effect=Timeout('offline'))
    def test_failed_sync_does_not_mark_stale_advisory_as_fresh(self, _request_get):
        advisory = ShellfishAdvisory.objects.create(
            bulletin_number='SB 22',
            bulletin_year=2026,
            title='Shellfish Bulletin No. 22',
            source_url='https://www.bfar.da.gov.ph/',
            pdf_url='https://www.bfar.da.gov.ph/advisory.pdf',
        )
        stale_checked_at = timezone.now() - timedelta(days=1)
        ShellfishAdvisory.objects.filter(pk=advisory.pk).update(
            last_checked_at=stale_checked_at,
        )

        from prediction.services.bfar import sync_latest_advisory
        result = sync_latest_advisory(force=True)

        self.assertIn('offline', result['error'])
        advisory.refresh_from_db()
        self.assertEqual(advisory.last_checked_at, stale_checked_at)


class SeededAdvisoryMigrationTests(TestCase):
    def test_empty_database_has_verified_cached_bulletin_for_initial_display(self):
        advisory = ShellfishAdvisory.objects.get(bulletin_number='SB 22')

        self.assertEqual(advisory.advisory_date, date(2026, 9, 16))
        self.assertIn('Dumanquillas Bay', advisory.affected_areas)
        self.assertEqual(advisory.verification_method, 'cached_snapshot')
        self.assertEqual(len(advisory.pdf_content), 814626)
        self.assertTrue(bytes(advisory.pdf_content).startswith(b'%PDF'))

    def test_seeded_pdf_endpoint_serves_inline_same_origin_copy(self):
        response = self.client.get(reverse('shellfish_advisory_pdf'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertNotIn('X-Frame-Options', response)
        self.assertIn(
            'https://sistem1-production.up.railway.app',
            response['Content-Security-Policy'],
        )
        self.assertEqual(len(response.content), 814626)
        self.assertTrue(response.content.startswith(b'%PDF'))
