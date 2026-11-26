import asyncio
import dataclasses
import datetime
import functools
import gzip
import json
import os
import pathlib
import re
import shutil
import time
import urllib.parse
import uuid

from absl import logging
import aiohttp
from aiohttp import web
from PIL import Image, ImageOps

import catalog
from common import ScanMode, ScanResult

LISTEN_HOST = '127.0.0.1'
LISTEN_PORT = 8787

ALLOWED_ORIGINS = frozenset({'https://nook.lol', 'https://www.nook.lol'})
EXPECTED_TURNSTILE_HOSTNAMES = frozenset({'nook.lol', 'www.nook.lol'})
EXPECTED_TURNSTILE_ACTION = 'web_scan'
EXPECTED_WEBSCAN_TURNSTILE_ACTION = 'webscan_submit'
TURNSTILE_VERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify'

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_WEBSCAN_SUBMIT_BYTES = 256 * 1024
MAX_WEBSCAN_EVENT_BYTES = 4 * 1024
MAX_IMAGE_FILES = 10
MAX_VIDEO_DURATION_SECONDS = 45.0
MIN_FRAME_HEIGHT = 472
MAX_QUEUE_DEPTH = 15
RATE_LIMIT_WINDOW_SECONDS = 600
RATE_LIMIT_MAX_UPLOADS = 10
WEBSCAN_SUBMIT_MAX_PER_WINDOW = 15
WEBSCAN_EVENT_MAX_PER_WINDOW = 10
SCAN_TIMEOUT_SECONDS = 90.0
JOB_TTL_SECONDS = 1800
WEB_CACHE_MAX_AGE_SECONDS = 14 * 24 * 3600
WEB_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024
FAILED_CACHE_MAX_AGE_SECONDS = 60 * 24 * 3600
FAILED_CACHE_MAX_BYTES = 1024 * 1024 * 1024

VIDEO_EXTS = frozenset({'.mp4', '.mov'})
IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp'})
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS
VALID_LOCALES = frozenset(catalog.LOCALE_MAP)
CONCRETE_LOCALES = frozenset(loc for loc in catalog.LOCALE_MAP if loc != 'auto')
VALID_WEBSCAN_MODES = frozenset({'catalog', 'recipes', 'critters', 'reactions', 'music', 'storage'})
MODE_MAX_ITEMS = {
    'catalog': 6200,
    'recipes': 1000,
    'critters': 200,
    'reactions': 88,
    'music': 110,
    'storage': 5000,
}
JOB_ID_RE = re.compile(r'^[a-f0-9]{16,32}$')
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x1f\x7f]')
_UNSAFE_SUBSTR_RE = re.compile(r'(?:https?://|[<>])', re.IGNORECASE)
_BASE_DIR = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass
class ScanJob:
    job_id: str
    client_ip: str
    scan_target: str
    cache_rel: str
    job_dir: pathlib.Path
    for_sale: bool
    created_at: float
    requested_locale: str = 'auto'
    status: str = 'queued'
    finished_at: float = 0.0
    url: str = ''
    count: int = 0
    mode: str = ''
    locale: str = ''
    error: str = ''


def _json_response(payload: dict, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status, headers={'Cache-Control': 'no-store'})


def _get_client_ip(request: web.Request) -> str:
    return (
        request.headers.get('CF-Connecting-IP')
        or request.remote
        or 'unknown'
    ).strip()


def _sanitize_filename(filename: str, fallback: str) -> str:
    base = pathlib.PurePosixPath(filename.replace('\\', '/')).name
    clean = re.sub(r'[^A-Za-z0-9._-]', '_', base).strip('._')
    return clean or fallback


def _prune_old_entries(root_dir: pathlib.Path, max_age_seconds: int, max_bytes: int | None = None) -> None:
    """Remove expired subdirectories/files and enforce an optional byte cap."""
    try:
        if not root_dir.exists():
            return
        now = time.time()
        kept = []
        for entry in root_dir.iterdir():
            try:
                stat = entry.stat()
            except OSError:
                continue
            if entry.is_dir():
                children = [p for p in entry.rglob('*') if p.is_file()]
                newest = max((p.stat().st_mtime for p in children), default=stat.st_mtime)
                size = sum(p.stat().st_size for p in children)
            else:
                newest = stat.st_mtime
                size = stat.st_size
            if (now - newest) > max_age_seconds:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
            else:
                kept.append((newest, size, entry))

        if max_bytes is not None:
            total = sum(size for _, size, _ in kept)
            if total > max_bytes:
                for _, size, entry in sorted(kept, key=lambda x: x[0]):
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink(missing_ok=True)
                    total -= size
                    if total <= max_bytes:
                        break
    except OSError:
        logging.warning('Failed to prune %s', root_dir, exc_info=True)


def _archive_failed_job(job_dir: pathlib.Path, failed_root: pathlib.Path, job_id: str, reason: str) -> None:
    """Move a failed upload directory into cache/failed/ for 60-day debugging retention."""
    try:
        failed_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        dest = failed_root / f'{stamp}_{job_id}'
        if job_dir.exists():
            shutil.move(str(job_dir), str(dest))
        else:
            dest.mkdir(parents=True, exist_ok=True)
        (dest / 'error.txt').write_text(f'{stamp} UTC\n{reason}\n', encoding='utf-8')
        _prune_old_entries(failed_root, FAILED_CACHE_MAX_AGE_SECONDS, FAILED_CACHE_MAX_BYTES)
    except OSError:
        logging.warning('Failed to archive failed upload %s', job_id, exc_info=True)


async def verify_turnstile(
    token: str,
    remoteip: str,
    secret: str,
    expected_actions: str | frozenset[str] = EXPECTED_TURNSTILE_ACTION,
) -> bool:
    """Verify a Cloudflare Turnstile response token fail-closed."""
    if not secret:
        return True
    if not token or len(token) > 2048:
        logging.warning('Turnstile verification rejected: missing or oversized token (len=%d).', len(token or ''))
        return False
    allowed_actions = {expected_actions} if isinstance(expected_actions, str) else set(expected_actions)
    body = urllib.parse.urlencode({
        'secret': secret,
        'response': token,
        'remoteip': remoteip,
    })
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                TURNSTILE_VERIFY_URL,
                data=body,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
            ) as resp:
                if resp.status != 200:
                    logging.warning('Turnstile siteverify returned HTTP %d.', resp.status)
                    return False
                data = await resp.json()
    except Exception:
        logging.warning('Turnstile siteverify request failed.', exc_info=True)
        return False

    ok = bool(
        data.get('success') is True
        and data.get('action') in allowed_actions
        and data.get('hostname') in EXPECTED_TURNSTILE_HOSTNAMES
    )
    if not ok:
        logging.warning(
            'Turnstile verification failed: success=%r error_codes=%r action=%r hostname=%r',
            data.get('success'),
            data.get('error-codes'),
            data.get('action'),
            data.get('hostname'),
        )
    return ok


@functools.lru_cache(maxsize=None)
def _get_canonical_items(mode: str, locale: str) -> frozenset[str]:
    """Return the closed-vocabulary allowlist of valid item names for (mode, locale)."""
    if mode == 'catalog':
        return frozenset(catalog._get_item_db(locale))
    if mode in ('recipes', 'critters', 'reactions', 'music'):
        with open(_BASE_DIR / mode / 'names.json', encoding='utf-8') as fp:
            names = [row[0] for row in json.load(fp)]
        if locale == 'en-us':
            return frozenset(names)
        with open(_BASE_DIR / mode / 'translations.json', encoding='utf-8') as fp:
            translations = json.load(fp)
        return frozenset(translations.get(n, {}).get(locale, n) for n in names)
    return frozenset()


@functools.lru_cache(maxsize=1)
def _load_storage_meta() -> dict:
    with gzip.open(_BASE_DIR / 'models' / 'acnh_storage_meta.json.gz', 'rb') as fp:
        return json.loads(fp.read().decode('utf-8'))


@functools.lru_cache(maxsize=None)
def _get_canonical_storage_triples(locale: str) -> frozenset[tuple[str, str, str]]:
    """Return valid (translated_name, variant, pattern) triples for Home Storage."""
    meta = _load_storage_meta()
    names = meta['locales'].get(locale) or meta['names']
    return frozenset((names[name_id], variant, pattern) for name_id, variant, pattern in meta['rows'])


def _sanitize_unmatched(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for entry in raw[:100]:
        if not isinstance(entry, str):
            continue
        s = _CONTROL_CHAR_RE.sub('', entry).strip()[:64]
        if not s or _UNSAFE_SUBSTR_RE.search(s):
            continue
        cleaned.append(s)
    return cleaned


def _validate_webscan_submission(payload: dict) -> tuple[ScanResult, dict]:
    """Validate a browser-native scan payload against canonical DB allowlists."""
    if not isinstance(payload, dict):
        raise ValueError('Invalid JSON payload.')

    mode = str(payload.get('mode') or '').strip().lower()
    if mode not in VALID_WEBSCAN_MODES:
        raise ValueError(f'Unsupported scan mode: {mode!r}')

    locale = str(payload.get('locale') or '').strip().lower()
    if locale not in CONCRETE_LOCALES:
        raise ValueError(f'Unsupported locale: {locale!r}')

    requested_locale = str(payload.get('requested_locale') or 'auto').strip().lower()
    if requested_locale not in VALID_LOCALES:
        requested_locale = 'auto'

    unmatched = _sanitize_unmatched(payload.get('unmatched'))
    details: list[dict] | None = None

    if mode == 'storage':
        raw_details = payload.get('details')
        if not isinstance(raw_details, list) or not raw_details:
            raise ValueError('Storage submission requires non-empty details.')
        if len(raw_details) > MODE_MAX_ITEMS['storage']:
            raise ValueError('Too many storage detail entries.')
        valid_triples = _get_canonical_storage_triples(locale)
        merged: dict[tuple[str, str, str], int] = {}
        total_count = 0
        for row in raw_details:
            if not isinstance(row, dict):
                raise ValueError('Invalid storage detail entry.')
            name = str(row.get('name') or '')
            variant = str(row.get('variant') or '')
            pattern = str(row.get('pattern') or '')
            try:
                cnt = int(row.get('count', 1))
            except (TypeError, ValueError) as exc:
                raise ValueError('Invalid storage item count.') from exc
            if cnt < 1 or cnt > 5000:
                raise ValueError('Storage item count out of range.')
            triple = (name, variant, pattern)
            if triple not in valid_triples:
                raise ValueError(f'Unknown storage item: {name!r}')
            total_count += cnt
            if total_count > 5000:
                raise ValueError('Total storage item count exceeds limit.')
            merged[triple] = merged.get(triple, 0) + cnt
        details = [
            {'name': n, 'variant': v, 'pattern': p, 'count': c}
            for (n, v, p), c in sorted(merged.items())
        ]
        items = sorted({d['name'] for d in details})
    else:
        raw_items = payload.get('items')
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError('Submission requires a non-empty items list.')
        if len(raw_items) > MODE_MAX_ITEMS[mode]:
            raise ValueError('Item count exceeds maximum for scan mode.')
        allowlist = _get_canonical_items(mode, locale)
        unique_items: set[str] = set()
        for entry in raw_items:
            if not isinstance(entry, str) or entry not in allowlist:
                raise ValueError(f'Unknown {mode} item: {entry!r}')
            unique_items.add(entry)
        items = sorted(unique_items)

    scan_result = ScanResult(
        mode=ScanMode[mode.upper()],
        items=items,
        locale=locale,
        unmatched=unmatched,
        details=details,
    )
    meta = {
        'requested_locale': requested_locale,
        'for_sale': bool(payload.get('for_sale', False)),
        'media_kind': 'image' if payload.get('media_kind') == 'image' else 'video',
        'file_bytes': max(0, min(100_000_000, int(payload.get('file_bytes') or 0))),
        'duration_ms': max(0, min(600_000, int(payload.get('duration_ms') or 0))),
    }
    return scan_result, meta


async def _probe_video(video_path: pathlib.Path) -> tuple[int, int, float]:
    """Return (width, height, duration_seconds) via ffprobe or raise ValueError."""
    proc = await asyncio.create_subprocess_exec(
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,codec_type:format=duration',
        '-of', 'json',
        str(video_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ValueError('ffprobe timed out')

    if proc.returncode != 0:
        raise ValueError('Invalid video file')

    try:
        meta = json.loads(stdout.decode('utf-8'))
        streams = meta.get('streams') or []
        if not streams:
            raise ValueError('No video stream found')
        width = int(streams[0].get('width') or 0)
        height = int(streams[0].get('height') or 0)
        duration = float((meta.get('format') or {}).get('duration') or 0.0)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid video metadata') from exc

    if width <= 0 or height <= 0:
        raise ValueError('Invalid video dimensions')
    return width, height, duration


def _validate_and_prepare_images(image_paths: list[pathlib.Path], job_dir: pathlib.Path) -> tuple[str, str]:
    """Validate screenshot heights and normalize multi-image uploads to frame_%d.png."""
    dims = []
    for path in image_paths:
        try:
            with Image.open(path) as im:
                im.load()
                width, height = im.size
        except Exception as exc:
            raise ValueError('Invalid image file.') from exc
        if height < MIN_FRAME_HEIGHT:
            raise AssertionError(f'Invalid resolution: {width}x{height}')
        dims.append((width, height))

    if len(image_paths) == 1:
        single = image_paths[0]
        if single.suffix.lower() == '.webp':
            out = job_dir / 'frame_0.png'
            with Image.open(single) as im:
                ImageOps.exif_transpose(im).convert('RGB').save(out, format='PNG')
            single.unlink(missing_ok=True)
            return str(out), out.name
        return str(single), single.name

    for idx, src in enumerate(image_paths):
        dst = job_dir / f'frame_{idx}.png'
        with Image.open(src) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            if rgb.size != (1280, 720):
                rgb = rgb.resize((1280, 720), Image.Resampling.LANCZOS)
            rgb.save(dst, format='PNG')
        if src.resolve() != dst.resolve():
            src.unlink(missing_ok=True)

    return str(job_dir / 'frame_%d.png'), 'frame_%d.png'


def create_app(
    *,
    wait_lock: asyncio.Lock,
    async_scan_fn,
    upload_fn,
    error_fn,
    turnstile_secret: str = '',
    cache_dir: pathlib.Path | str = 'cache',
    on_webscan_event=None,
) -> web.Application:
    cache_root = pathlib.Path(cache_dir)
    web_cache_root = cache_root / 'web'
    failed_cache_root = cache_root / 'failed'

    jobs: dict[str, ScanJob] = {}
    ip_timestamps: dict[str, list[float]] = {}
    webscan_submit_timestamps: dict[str, list[float]] = {}
    webscan_event_timestamps: dict[str, list[float]] = {}

    def _check_rate_limit(bucket: dict[str, list[float]], client_ip: str, max_count: int) -> bool:
        now = time.time()
        # Periodically prune idle IP keys so memory stays bounded.
        if len(bucket) > 512:
            stale_ips = [
                ip for ip, ts_list in bucket.items()
                if not ts_list or (now - ts_list[-1]) >= RATE_LIMIT_WINDOW_SECONDS
            ]
            for ip in stale_ips:
                bucket.pop(ip, None)
        recent = [ts for ts in bucket.get(client_ip, []) if (now - ts) < RATE_LIMIT_WINDOW_SECONDS]
        if recent:
            bucket[client_ip] = recent
        else:
            bucket.pop(client_ip, None)
        return len(recent) < max_count

    def _record_rate_limit(bucket: dict[str, list[float]], client_ip: str) -> None:
        bucket.setdefault(client_ip, []).append(time.time())

    def _prune_jobs() -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        stale = [
            jid for jid, job in jobs.items()
            if job.status in ('done', 'error') and job.finished_at and job.finished_at < cutoff
        ]
        for jid in stale:
            jobs.pop(jid, None)

    def _queued_job_ids() -> list[str]:
        return [jid for jid, job in jobs.items() if job.status == 'queued']

    def _queue_position(job_id: str) -> int:
        ahead = 0
        for jid, job in jobs.items():
            if jid == job_id:
                break
            if job.status == 'queued':
                ahead += 1
        scanning_offset = 1 if wait_lock.locked() else 0
        return 1 + ahead + scanning_offset

    async def _run_job(job: ScanJob) -> None:
        try:
            async with wait_lock:
                job.status = 'scanning'
                try:
                    result = await asyncio.wait_for(
                        async_scan_fn(job.scan_target, locale=job.requested_locale, for_sale=job.for_sale),
                        timeout=SCAN_TIMEOUT_SECONDS,
                    )
                except AssertionError as exc:
                    msg = error_fn(str(exc), source='web')
                    _archive_failed_job(job.job_dir, failed_cache_root, job.job_id, msg)
                    job.status = 'error'
                    job.error = msg
                    return
                except asyncio.TimeoutError:
                    msg = 'Scan timed out. Please trim your video or split large catalogs into subsections.'
                    _archive_failed_job(job.job_dir, failed_cache_root, job.job_id, msg)
                    job.status = 'error'
                    job.error = msg
                    return
                except Exception as exc:
                    logging.exception('Unexpected web scan error on %s', job.job_id)
                    msg = 'Failed to scan media. Make sure the file was exported directly from your Nintendo Switch.'
                    _archive_failed_job(job.job_dir, failed_cache_root, job.job_id, f'{msg}\n{exc!r}')
                    job.status = 'error'
                    job.error = msg
                    return

                if not result.items:
                    msg = 'Did not find any items in your upload.'
                    _archive_failed_job(job.job_dir, failed_cache_root, job.job_id, msg)
                    job.status = 'error'
                    job.error = msg
                    return

                try:
                    entity = await asyncio.to_thread(upload_fn, result, None, job.cache_rel, source='web')
                except Exception:
                    logging.exception('Failed to upload web scan results to Datastore for %s', job.job_id)
                    _archive_failed_job(job.job_dir, failed_cache_root, job.job_id, 'Failed to upload results to Datastore.')
                    job.status = 'error'
                    job.error = 'Found items, but failed to save results. Please try again.'
                    return

                job.url = f"https://nook.lol/{entity['hash']}"
                job.count = len(result.items)
                job.mode = result.mode.name.lower()
                job.locale = result.locale
                job.status = 'done'
                _prune_old_entries(web_cache_root, WEB_CACHE_MAX_AGE_SECONDS, WEB_CACHE_MAX_BYTES)
        finally:
            if job.status not in ('done', 'error'):
                job.status = 'error'
                job.error = job.error or 'Unexpected error during scan.'
            job.finished_at = time.time()

    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        origin = request.headers.get('Origin')
        if request.method == 'OPTIONS':
            resp = web.Response(status=204)
        else:
            resp = await handler(request)
        if origin in ALLOWED_ORIGINS:
            resp.headers['Access-Control-Allow-Origin'] = origin
            resp.headers['Vary'] = 'Origin'
            resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            resp.headers['Access-Control-Max-Age'] = '86400'
        resp.headers.setdefault('Cache-Control', 'no-store')
        return resp

    async def handle_health(_request: web.Request) -> web.Response:
        _prune_jobs()
        return _json_response({
            'ok': True,
            'queued': len(_queued_job_ids()),
            'busy': wait_lock.locked(),
        })

    async def handle_status(request: web.Request) -> web.Response:
        _prune_jobs()
        job_id = request.match_info.get('job_id', '')
        if not JOB_ID_RE.match(job_id) or job_id not in jobs:
            return _json_response({'error': 'Scan job not found or expired.'}, status=404)
        job = jobs[job_id]
        payload = {
            'job_id': job.job_id,
            'status': job.status,
            'position': _queue_position(job.job_id) if job.status == 'queued' else 0,
            'url': job.url,
            'count': job.count,
            'mode': job.mode,
            'locale': job.locale,
            'error': job.error,
        }
        return _json_response(payload)

    async def handle_scan(request: web.Request) -> web.Response:
        _prune_jobs()
        origin = request.headers.get('Origin')
        if origin not in ALLOWED_ORIGINS:
            return _json_response({'error': 'Origin not allowed.'}, status=403)

        client_ip = _get_client_ip(request)
        if len(_queued_job_ids()) >= MAX_QUEUE_DEPTH:
            return _json_response(
                {'error': 'The scanner queue is currently full. Please try again in a minute.'},
                status=503,
            )

        if any(j.client_ip == client_ip and j.status in ('queued', 'scanning') for j in jobs.values()):
            return _json_response(
                {'error': 'You already have a scan in progress. Please wait for it to finish.'},
                status=429,
            )

        now = time.time()
        recent = [ts for ts in ip_timestamps.get(client_ip, []) if (now - ts) < RATE_LIMIT_WINDOW_SECONDS]
        ip_timestamps[client_ip] = recent
        if len(recent) >= RATE_LIMIT_MAX_UPLOADS:
            return _json_response(
                {'error': f'Rate limit reached (max {RATE_LIMIT_MAX_UPLOADS} scans per 10 minutes). Please wait a few minutes and try again.'},
                status=429,
            )

        if request.content_length is not None and request.content_length > (MAX_UPLOAD_BYTES + 1024 * 1024):
            return _json_response(
                {'error': 'Upload exceeds the 50 MB limit. Trim your video or split your catalog into subsections.'},
                status=413,
            )

        if not (request.content_type or '').startswith('multipart/'):
            return _json_response({'error': 'Expected multipart form upload.'}, status=400)

        job_id = uuid.uuid4().hex[:24]
        job_dir = web_cache_root / job_id
        for_sale = False
        requested_locale = 'auto'
        turnstile_token = ''
        turnstile_checked = not bool(turnstile_secret)
        turnstile_ok = not bool(turnstile_secret)
        saved_paths: list[pathlib.Path] = []
        media_kind = ''
        total_bytes = 0

        try:
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break

                if part.name == 'cf-turnstile-response':
                    raw = await part.read(decode=False)
                    turnstile_token = raw.decode('utf-8', 'replace').strip()
                    if turnstile_secret:
                        turnstile_checked = True
                        turnstile_ok = await verify_turnstile(turnstile_token, client_ip, turnstile_secret)
                        if not turnstile_ok:
                            shutil.rmtree(job_dir, ignore_errors=True)
                            return _json_response(
                                {'error': 'Bot verification failed. Please refresh and try again.'},
                                status=403,
                            )
                elif part.name == 'for_sale':
                    raw = await part.read(decode=False)
                    for_sale = raw.decode('utf-8', 'replace').strip().lower() in ('1', 'true', 'yes', 'on')
                elif part.name == 'locale':
                    raw = await part.read(decode=False)
                    loc = raw.decode('utf-8', 'replace').strip().lower() or 'auto'
                    if loc not in VALID_LOCALES:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        return _json_response({'error': f'Unsupported language: {loc}'}, status=400)
                    requested_locale = loc
                elif part.name in ('files', 'file'):
                    if turnstile_secret and not turnstile_ok:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        return _json_response(
                            {'error': 'Bot verification failed. Please refresh and try again.'},
                            status=403,
                        )
                    if not part.filename:
                        continue
                    safe_name = _sanitize_filename(part.filename, f'upload_{len(saved_paths)}')
                    ext = pathlib.Path(safe_name).suffix.lower()
                    if ext not in ALLOWED_EXTS:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        return _json_response(
                            {'error': 'Unsupported file type. Upload an MP4/MOV video or JPG/PNG/WebP screenshots.'},
                            status=400,
                        )

                    part_kind = 'video' if ext in VIDEO_EXTS else 'image'
                    if media_kind and media_kind != part_kind:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        return _json_response(
                            {'error': 'Upload either one video or screenshots, not both.'},
                            status=400,
                        )
                    media_kind = part_kind
                    if media_kind == 'video' and saved_paths:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        return _json_response(
                            {'error': 'Only one video file can be scanned at a time.'},
                            status=400,
                        )
                    if media_kind == 'image' and len(saved_paths) >= MAX_IMAGE_FILES:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        return _json_response(
                            {'error': f'Too many screenshots (max {MAX_IMAGE_FILES} per scan).'},
                            status=400,
                        )

                    job_dir.mkdir(parents=True, exist_ok=True)
                    dest = job_dir / f'{len(saved_paths):02d}_{safe_name}'
                    with open(dest, 'wb') as out_fp:
                        while True:
                            chunk = await part.read_chunk(65536)
                            if not chunk:
                                break
                            total_bytes += len(chunk)
                            if total_bytes > MAX_UPLOAD_BYTES:
                                out_fp.close()
                                shutil.rmtree(job_dir, ignore_errors=True)
                                return _json_response(
                                    {'error': 'Upload exceeds the 50 MB limit. Trim your video or split your catalog into subsections.'},
                                    status=413,
                                )
                            out_fp.write(chunk)
                    saved_paths.append(dest)
        except web.HTTPRequestEntityTooLarge:
            shutil.rmtree(job_dir, ignore_errors=True)
            return _json_response(
                {'error': 'Upload exceeds the 50 MB limit. Trim your video or split your catalog into subsections.'},
                status=413,
            )
        except ConnectionError:
            shutil.rmtree(job_dir, ignore_errors=True)
            logging.info('Client disconnected during multipart upload.')
            return _json_response({'error': 'Upload interrupted.'}, status=400)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            logging.warning('Failed to read multipart upload.', exc_info=True)
            return _json_response({'error': 'Failed to read uploaded file.'}, status=400)

        if turnstile_secret and (not turnstile_checked or not turnstile_ok):
            turnstile_ok = await verify_turnstile(turnstile_token, client_ip, turnstile_secret)
            if not turnstile_ok:
                shutil.rmtree(job_dir, ignore_errors=True)
                return _json_response(
                    {'error': 'Bot verification failed. Please refresh and try again.'},
                    status=403,
                )

        if not saved_paths:
            shutil.rmtree(job_dir, ignore_errors=True)
            return _json_response({'error': 'Please choose a video or screenshot to scan.'}, status=400)

        try:
            if media_kind == 'video':
                width, height, duration = await _probe_video(saved_paths[0])
                if height < MIN_FRAME_HEIGHT:
                    raise AssertionError(f'Invalid resolution: {width}x{height}')
                if duration > MAX_VIDEO_DURATION_SECONDS:
                    raise ValueError('Video is too long (max 45 seconds). Trim the clip around the start and end of scrolling.')
                scan_target = str(saved_paths[0])
                cache_rel = f'web/{job_id}/{saved_paths[0].name}'
            else:
                loop = asyncio.get_running_loop()
                scan_target, rel_leaf = await loop.run_in_executor(
                    None,
                    functools.partial(_validate_and_prepare_images, saved_paths, job_dir),
                )
                cache_rel = f'web/{job_id}/{rel_leaf}'
        except AssertionError as exc:
            msg = error_fn(str(exc), source='web')
            _archive_failed_job(job_dir, failed_cache_root, job_id, msg)
            return _json_response({'error': msg}, status=400)
        except ValueError as exc:
            msg = str(exc)
            _archive_failed_job(job_dir, failed_cache_root, job_id, msg)
            return _json_response({'error': msg}, status=400)

        ip_timestamps[client_ip].append(time.time())
        job = ScanJob(
            job_id=job_id,
            client_ip=client_ip,
            scan_target=scan_target,
            cache_rel=cache_rel,
            job_dir=job_dir,
            for_sale=for_sale,
            created_at=time.time(),
            requested_locale=requested_locale,
        )
        jobs[job_id] = job
        pos = _queue_position(job_id)
        asyncio.create_task(_run_job(job))
        return _json_response({'job_id': job_id, 'status': 'queued', 'position': pos}, status=202)

    async def handle_webscan_submit(request: web.Request) -> web.Response:
        origin = request.headers.get('Origin')
        if origin not in ALLOWED_ORIGINS:
            return _json_response({'error': 'Origin not allowed.'}, status=403)

        client_ip = _get_client_ip(request)
        if not _check_rate_limit(webscan_submit_timestamps, client_ip, WEBSCAN_SUBMIT_MAX_PER_WINDOW):
            return _json_response(
                {'error': f'Rate limit reached (max {WEBSCAN_SUBMIT_MAX_PER_WINDOW} saves per 10 minutes).'},
                status=429,
            )

        if request.content_length is not None and request.content_length > MAX_WEBSCAN_SUBMIT_BYTES:
            return _json_response({'error': 'Payload too large.'}, status=413)

        raw_body = await request.read()
        if len(raw_body) > MAX_WEBSCAN_SUBMIT_BYTES:
            return _json_response({'error': 'Payload too large.'}, status=413)

        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _json_response({'error': 'Invalid JSON body.'}, status=400)

        if turnstile_secret:
            token = str((payload or {}).get('turnstile_token') or '').strip()
            ok = await verify_turnstile(
                token, client_ip, turnstile_secret,
                expected_actions=frozenset({EXPECTED_TURNSTILE_ACTION, EXPECTED_WEBSCAN_TURNSTILE_ACTION}),
            )
            if not ok:
                return _json_response({'error': 'Bot verification failed. Please refresh and try again.'}, status=403)

        try:
            scan_result, meta = _validate_webscan_submission(payload)
        except ValueError as exc:
            return _json_response({'error': str(exc)}, status=400)

        _record_rate_limit(webscan_submit_timestamps, client_ip)
        try:
            entity = await asyncio.to_thread(upload_fn, scan_result, None, 'web-native', source='webscan')
        except Exception:
            logging.exception('Failed to save browser-native scan to Datastore.')
            return _json_response({'error': 'Failed to save results. Please try again.'}, status=500)

        key_hash = str(entity['hash'])
        url = f'https://nook.lol/{key_hash}'
        mode_str = scan_result.mode.name.lower()
        unmatched = scan_result.unmatched or []
        logging.info(
            'WEB_NATIVE_SCAN hash=%s mode=%s locale=%s items=%d unmatched=%d duration_ms=%d',
            key_hash, mode_str, scan_result.locale, len(scan_result.items), len(unmatched), meta['duration_ms'],
        )
        if on_webscan_event is not None:
            try:
                on_webscan_event({
                    'source': 'webscan',
                    'status': 'ok',
                    'scan_type': mode_str,
                    'locale': scan_result.locale,
                    'requested_locale': meta['requested_locale'],
                    'for_sale': meta['for_sale'],
                    'media_kind': meta['media_kind'],
                    'file_bytes': meta['file_bytes'],
                    'duration_ms': meta['duration_ms'],
                    'item_count': len(scan_result.items),
                    'unmatched_count': len(unmatched),
                    'unmatched_text': '\n'.join(unmatched),
                    'result_hash': key_hash,
                    'client_ip': client_ip,
                })
            except Exception:
                logging.warning('on_webscan_event failed on submit.', exc_info=True)

        return _json_response({
            'hash': key_hash,
            'url': url,
            'count': len(scan_result.items),
            'mode': mode_str,
            'locale': scan_result.locale,
        })

    async def handle_webscan_event(request: web.Request) -> web.Response:
        origin = request.headers.get('Origin')
        if origin not in ALLOWED_ORIGINS:
            return _json_response({'error': 'Origin not allowed.'}, status=403)

        client_ip = _get_client_ip(request)
        if not _check_rate_limit(webscan_event_timestamps, client_ip, WEBSCAN_EVENT_MAX_PER_WINDOW):
            return _json_response({'error': 'Rate limit reached.'}, status=429)

        raw_body = await request.read()
        if len(raw_body) > MAX_WEBSCAN_EVENT_BYTES:
            return _json_response({'error': 'Payload too large.'}, status=413)
        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _json_response({'error': 'Invalid JSON body.'}, status=400)
        if not isinstance(payload, dict):
            return _json_response({'error': 'Invalid JSON body.'}, status=400)

        status = str(payload.get('status') or 'error').strip().lower()
        if status not in ('rejected', 'error'):
            status = 'error'
        mode = str(payload.get('mode') or '').strip().lower()
        if mode not in VALID_WEBSCAN_MODES:
            mode = ''
        req_locale = str(payload.get('requested_locale') or 'auto').strip().lower()
        if req_locale not in VALID_LOCALES:
            req_locale = 'auto'
        raw_err = _CONTROL_CHAR_RE.sub(' ', str(payload.get('error_message') or '')).strip()[:200]
        if _UNSAFE_SUBSTR_RE.search(raw_err):
            raw_err = 'Browser scan error'
        duration_ms = max(0, min(600_000, int(payload.get('duration_ms') or 0)))
        file_bytes = max(0, min(100_000_000, int(payload.get('file_bytes') or 0)))
        media_kind = 'image' if payload.get('media_kind') == 'image' else 'video'

        _record_rate_limit(webscan_event_timestamps, client_ip)
        logging.info(
            'WEB_NATIVE_EVENT status=%s mode=%s duration_ms=%d error=%r',
            status, mode or '-', duration_ms, raw_err,
        )
        if on_webscan_event is not None:
            try:
                on_webscan_event({
                    'source': 'webscan',
                    'status': status,
                    'scan_type': mode,
                    'locale': '',
                    'requested_locale': req_locale,
                    'for_sale': False,
                    'media_kind': media_kind,
                    'file_bytes': file_bytes,
                    'duration_ms': duration_ms,
                    'item_count': 0,
                    'unmatched_count': 0,
                    'error_message': raw_err,
                    'client_ip': client_ip,
                })
            except Exception:
                logging.warning('on_webscan_event failed on event.', exc_info=True)
        return _json_response({'ok': True})

    app = web.Application(middlewares=[cors_middleware], client_max_size=MAX_UPLOAD_BYTES + 2 * 1024 * 1024)
    app.router.add_get('/api/health', handle_health)
    app.router.add_get('/api/status/{job_id}', handle_status)
    app.router.add_post('/api/scan', handle_scan)
    app.router.add_post('/api/webscan/submit', handle_webscan_submit)
    app.router.add_post('/api/webscan/event', handle_webscan_event)
    return app


async def start_server(
    *,
    wait_lock: asyncio.Lock,
    async_scan_fn,
    upload_fn,
    error_fn,
    turnstile_secret: str = '',
    host: str = LISTEN_HOST,
    port: int = LISTEN_PORT,
    cache_dir: pathlib.Path | str = 'cache',
    on_webscan_event=None,
) -> web.AppRunner:
    """Start the embedded aiohttp server on the running asyncio loop."""
    app = create_app(
        wait_lock=wait_lock,
        async_scan_fn=async_scan_fn,
        upload_fn=upload_fn,
        error_fn=error_fn,
        turnstile_secret=turnstile_secret,
        cache_dir=cache_dir,
        on_webscan_event=on_webscan_event,
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logging.info('Web upload API listening on http://%s:%d', host, port)
    return runner
