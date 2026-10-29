import json
import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from django.utils import timezone
from PIL import Image
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import ShellfishAdvisory
from .scheduler import schedule_sync

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_PATH = MODEL_DIR / 'shell_model8.tflite'
LABELS_PATH = MODEL_DIR / 'labels.json'
IMAGE_SHAPE = (224, 224)

# Images directory for saving captured images
IMAGES_DIR = Path(__file__).resolve().parent / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

@csrf_exempt
@require_http_methods(["GET", "POST"])
def shellfish_advisory(request):
    """
    Return the cached BFAR advisory and queue a background refresh if needed.
    
    GET: Returns cached data immediately and refreshes stale data in the background
    POST: Queues a forced background refresh
    
    Response:
    - 200: Cached advisory data is available
    - 202: No cached advisory is available; initial sync is queued
    - 503: No cached advisory is available and sync could not be queued
    """
    try:
        sync_result = schedule_sync(force=request.method == 'POST')
    except Exception as exc:
        logger.exception('Unable to queue BFAR advisory sync')
        sync_result = {
            'attempted': False,
            'changed': False,
            'cached': False,
            'queued': False,
            'error': str(exc),
        }
    advisory = ShellfishAdvisory.objects.defer('pdf_content').first()
    sync_result['cached'] = bool(advisory)
    sync_result['stale'] = bool(
        advisory and timezone.now() - advisory.last_checked_at >= timedelta(hours=6)
    )
    if sync_result['stale']:
        sync_result['warning'] = (
            'Showing a cached advisory that may be outdated. '
            'Check the official BFAR source for current conditions.'
        )

    # Parse affected_areas from newline-separated string to list
    affected_areas = []
    if advisory and advisory.affected_areas:
        affected_areas = [area.strip() for area in advisory.affected_areas.split('\n') if area.strip()]

    if advisory is None:
        logger.warning(f"No advisory available. Sync result: {sync_result}")
        status = (
            503 if sync_result.get('error')
            else 202 if sync_result.get('queued')
            else 200
        )
        return JsonResponse({
            'advisory': None,
            'last_updated': None,
            'sync': sync_result,
        }, status=status)

    logger.info(
        "Returning latest advisory: bulletin=%s date=%s source=%s method=%s",
        advisory.bulletin_number,
        advisory.advisory_date.isoformat() if advisory.advisory_date else None,
        advisory.source_url,
        'forced' if request.method == 'POST' else 'cached',
    )

    response_data = {
        'advisory': {
            'bulletin_number': advisory.bulletin_number,
            'title': advisory.title,
            'date': advisory.advisory_date.isoformat() if advisory.advisory_date else None,
            'status': advisory.status,
            'affected_areas': affected_areas,
            'important_information': advisory.important_information,
            'source_url': advisory.source_url,
            'pdf_url': advisory.pdf_url,
            'pdf_filename': advisory.pdf_filename,
            'pdf_content_type': advisory.pdf_content_type,
            'pdf_size': advisory.pdf_size,
            'stale': sync_result['stale'],
            'warning': sync_result.get('warning'),
        },
        'last_updated': advisory.last_checked_at.isoformat(),
        'sync': sync_result,
    }
    
    # Return 200 if we have advisory data (fresh or cached)
    # Only return 503 if sync failed AND no cached advisory exists
    status = 200
    
    return JsonResponse(response_data, status=status)


@require_http_methods(["GET"])
def shellfish_advisory_pdf(request):
    """
    Serve a locally cached advisory PDF, or redirect directly to BFAR.
    """
    bulletin_number = request.GET.get('bulletin_number')
    bulletin_year = request.GET.get('year')
    if bulletin_number:
        filters = {'bulletin_number': bulletin_number}
        if bulletin_year:
            filters['bulletin_year'] = bulletin_year
        advisory = ShellfishAdvisory.objects.defer('pdf_content').filter(**filters).first()
    else:
        advisory = ShellfishAdvisory.objects.defer('pdf_content').first()
    if advisory is None or not advisory.pdf_url:
        return JsonResponse(
            {'error': 'No cached advisory PDF is available.'},
            status=404
        )

    filename = advisory.pdf_filename or 'bfar-shellfish-advisory.pdf'
    if advisory.pdf_content:
        proxy_response = HttpResponse(
            bytes(advisory.pdf_content),
            content_type='application/pdf',
        )
        proxy_response['Content-Disposition'] = f'inline; filename="{filename}"'
        proxy_response['Cache-Control'] = 'private, max-age=300'
        return proxy_response

    pdf_url = urlsplit(advisory.pdf_url)
    if (
        pdf_url.scheme != 'https'
        or pdf_url.hostname not in {'bfar.da.gov.ph', 'www.bfar.da.gov.ph'}
    ):
        logger.error('Refusing to redirect to non-BFAR PDF URL: %s', advisory.pdf_url)
        return JsonResponse(
            {'error': 'The cached advisory PDF URL is not an official BFAR URL.'},
            status=502
        )

    redirect_response = HttpResponseRedirect(advisory.pdf_url)
    redirect_response['Cache-Control'] = 'no-store'
    return redirect_response


def _ensure_probabilities(output):
    """
    Ensure output is normalized to probabilities [0, 1] that sum to 1.
    
    Models trained with EfficientNetV2 and TFLite may output:
    1. Logits (unbounded, can be negative) → apply softmax
    2. Pre-softmax probabilities (non-negative, sum ≈ 1.0) → use directly
    
    Heuristic: If all values are non-negative AND sum to ~1.0, treat as probabilities.
    Otherwise, treat as logits and apply softmax.
    """
    output = np.asarray(output, dtype=np.float32).reshape(-1)
    if output.size == 0:
        raise ValueError('Model returned an empty prediction vector.')

    # Check if output looks like probabilities
    all_non_negative = np.all(output >= 0)
    output_sum = np.sum(output)
    looks_like_probabilities = all_non_negative and (0.99 <= output_sum <= 1.01)
    
    if looks_like_probabilities:
        # Already probabilities, just normalize slightly to ensure exact 1.0
        return output / output_sum
    
    # Otherwise, treat as logits and apply softmax
    shifted_output = output - np.max(output)
    exp_values = np.exp(shifted_output)
    total = np.sum(exp_values)
    if total <= 0:
        raise ValueError('Model output could not be normalized into probabilities.')
    
    return exp_values / total


@lru_cache(maxsize=1)
def get_model_and_labels():
    try:
        from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
    except ImportError as exc:  # pragma: no cover - runtime env issue
        raise RuntimeError(f'LiteRT runtime is missing or broken: {exc}') from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')

    if not LABELS_PATH.exists():
        raise FileNotFoundError(f'Labels file not found: {LABELS_PATH}')

    with LABELS_PATH.open('r', encoding='utf-8') as labels_file:
        labels = json.load(labels_file)

    interpreter = LiteRTInterpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter, labels


def _preprocess_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize(IMAGE_SHAPE, Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


@csrf_exempt
@require_http_methods(["POST"])
def predict_shell(request):
    uploaded_file = request.FILES.get('image')

    if uploaded_file is None:
        return JsonResponse({
            'error': 'No image uploaded.',
            'prediction': 'unknown',
            'confidence': 0.0,
            'label': 'unknown',
        }, status=400)

    file_name = uploaded_file.name.lower()
    if not file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        return JsonResponse({
            'error': 'Unsupported image type.',
            'prediction': 'unknown',
            'confidence': 0.0,
            'label': 'unknown',
        }, status=400)

    try:
        interpreter, labels = get_model_and_labels()
        processed_image = _preprocess_image(uploaded_file)

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        interpreter.set_tensor(input_details['index'], processed_image.astype(np.float32))
        interpreter.invoke()

        output_tensor = interpreter.get_tensor(output_details['index'])
        logits = np.asarray(output_tensor, dtype=np.float32).reshape(-1)
        probabilities = _ensure_probabilities(logits)

        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

        if class_index < 0 or class_index >= len(labels):
            raise ValueError(f'Invalid model output index: {class_index}')

        prediction = str(labels[class_index])
        label = prediction.replace('_', ' ').replace('-', ' ').title()
        probabilities_map = {
            str(labels[i]): float(probabilities[i])
            for i in range(len(labels))
        }

        # Save the uploaded image to images directory
        uploaded_file.seek(0)
        saved_filename = f"{Path(uploaded_file.name).stem}_{class_index}_{int(confidence * 100)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        saved_path = IMAGES_DIR / saved_filename
        
        with open(saved_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        return JsonResponse({
            'prediction': prediction,
            'confidence': float(confidence),
            'label': label,
            'probabilities': probabilities_map,
            'message': 'Prediction received successfully.',
            'filename': uploaded_file.name,
            'content_type': uploaded_file.content_type,
            'saved_image': saved_filename,
        }, status=200)

    except Exception as exc:
        return JsonResponse({
            'error': f'Prediction failed: {exc}',
            'prediction': 'unknown',
            'confidence': 0.0,
            'label': 'unknown',
        }, status=500)

@require_http_methods(["GET"])
def shellfish_advisory_history(request):
    """
    Get paginated historical BFAR shellfish bulletins.
    
    Query parameters:
    - page: Page number (default 1, 1-indexed)
    - page_size: Entries per page (default 5, max 50)
    
    Response:
    - 200: Returns paginated list of bulletins, newest first
    - 400: Invalid page/page_size parameters
    """
    try:
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 5))
        
        if page < 1:
            return JsonResponse(
                {'error': 'page must be >= 1'},
                status=400
            )
        
        if page_size < 1 or page_size > 50:
            return JsonResponse(
                {'error': 'page_size must be between 1 and 50'},
                status=400
            )
    except ValueError:
        return JsonResponse(
            {'error': 'page and page_size must be integers'},
            status=400
        )
    
    requested_year = request.GET.get('year', 'current')
    if requested_year == 'current':
        selected_year = timezone.localdate().year
    else:
        try:
            selected_year = int(requested_year)
        except ValueError:
            return JsonResponse({'error': 'year must be current or a four-digit year'}, status=400)
        if selected_year < 2000 or selected_year > 2100:
            return JsonResponse({'error': 'year must be between 2000 and 2100'}, status=400)

    # Filter by year, then sort by the authoritative advisory date.
    all_bulletins = ShellfishAdvisory.objects.defer('pdf_content').filter(
        bulletin_year=selected_year,
    ).order_by('-advisory_date', '-first_seen_at')
    total_count = all_bulletins.count()
    
    # Calculate pagination
    total_pages = (total_count + page_size - 1) // page_size
    if page > total_pages and total_count > 0:
        return JsonResponse(
            {'error': f'page must be <= {total_pages}'},
            status=400
        )
    
    # Get page data
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_bulletins = all_bulletins[start_idx:end_idx]
    
    # Format bulletins for response
    bulletins = []
    for bulletin in page_bulletins:
        bulletins.append({
            'bulletin_number': bulletin.bulletin_number,
            'year': bulletin.bulletin_year,
            'date': bulletin.advisory_date.isoformat() if bulletin.advisory_date else None,
            'title': bulletin.title,
            'pdf_url': bulletin.pdf_url,
            'source_url': bulletin.source_url,
            'pdf_filename': bulletin.pdf_filename,
            'pdf_content_type': bulletin.pdf_content_type,
            'pdf_size': bulletin.pdf_size,
            'pdf_text_extracted': bulletin.pdf_text_extracted,
            'verification_method': bulletin.verification_method,
        })
    
    logger.info(f"Returning history page {page}/{total_pages} with {len(bulletins)} bulletins")
    
    return JsonResponse({
        'bulletins': bulletins,
        'pagination': {
            'page': page,
            'page_size': page_size,
            'total': total_count,
            'year': selected_year,
            'total_pages': total_pages,
            'has_previous': page > 1,
            'has_next': page < total_pages,
        }
    }, status=200)
