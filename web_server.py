import asyncio
import dataclasses
import datetime
import functools
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

LISTEN_HOST = '127.0.0.1'
LISTEN_PORT = 8787

ALLOWED_ORIGINS = frozenset({'https://nook.lol', 'https://www.nook.lol'})
EXPECTED_TURNSTILE_HOSTNAMES = frozenset({'nook.lol', 'www.nook.lol'})
EXPECTED_TURNSTILE_ACTION = 'web_scan'
TURNSTILE_VERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify'

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_FILES = 10
MAX_VIDEO_DURATION_SECONDS = 45.0
MIN_FRAME_HEIGHT = 472
MAX_QUEUE_DEPTH = 15
RATE_LIMIT_WINDOW_SECONDS = 600
RATE_LIMIT_MAX_UPLOADS = 5
SCAN_TIMEOUT_SECONDS = 90.0
JOB_TTL_SECONDS = 1800
WEB_CACHE_MAX_AGE_SECONDS = 14 * 24 * 3600
FAILED_CACHE_MAX_AGE_SECONDS = 60 * 24 * 3600
FAILED_CACHE_MAX_BYTES = 1024 * 1024 * 1024

VIDEO_EXTS = frozenset({'.mp4', '.mov'})
IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp'})
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS
VALID_LOCALES = frozenset(catalog.LOCALE_MAP)
JOB_ID_RE = re.compile(r'^[a-f0-9]{16,32}$')


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


async def verify_turnstile(token: str, remoteip: str, secret: str) -> bool:
    """Verify a Cloudflare Turnstile response token fail-closed."""
    if not secret:
        return True
    if not token or len(token) > 2048:
        return False
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
                    return False
                data = await resp.json()
    except Exception:
        logging.warning('Turnstile siteverify request failed.', exc_info=True)
        return False

    return bool(
        data.get('success') is True
        and data.get('action') == EXPECTED_TURNSTILE_ACTION
        and data.get('hostname') in EXPECTED_TURNSTILE_HOSTNAMES
    )


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
) -> web.Application:
    cache_root = pathlib.Path(cache_dir)
    web_cache_root = cache_root / 'web'
    failed_cache_root = cache_root / 'failed'

    jobs: dict[str, ScanJob] = {}
    ip_timestamps: dict[str, list[float]] = {}

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
                    entity = upload_fn(result, None, job.cache_rel, source='web')
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
                _prune_old_entries(web_cache_root, WEB_CACHE_MAX_AGE_SECONDS)
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
                {'error': 'Rate limit reached (max 5 scans per 10 minutes). Please wait a few minutes and try again.'},
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

    app = web.Application(middlewares=[cors_middleware], client_max_size=MAX_UPLOAD_BYTES + 2 * 1024 * 1024)
    app.router.add_get('/api/health', handle_health)
    app.router.add_get('/api/status/{job_id}', handle_status)
    app.router.add_post('/api/scan', handle_scan)
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
) -> web.AppRunner:
    """Start the embedded aiohttp server on the running asyncio loop."""
    app = create_app(
        wait_lock=wait_lock,
        async_scan_fn=async_scan_fn,
        upload_fn=upload_fn,
        error_fn=error_fn,
        turnstile_secret=turnstile_secret,
        cache_dir=cache_dir,
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logging.info('Web upload API listening on http://%s:%d', host, port)
    return runner
