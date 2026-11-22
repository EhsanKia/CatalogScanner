import asyncio
import concurrent.futures
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import aiohttp
from aiohttp.test_utils import AioHTTPTestCase
from PIL import Image

import discord_bot
import scanner
import web_server

ALLOWED_ORIGIN = {'Origin': 'https://nook.lol', 'CF-Connecting-IP': '203.0.113.10'}


def _png_bytes(width: int, height: int, color=(80, 160, 220)) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), color).save(buf, format='PNG')
    return buf.getvalue()


def _multipart_upload(fields=None, files=None) -> aiohttp.FormData:
    form = aiohttp.FormData()
    for name, val in (fields or []):
        form.add_field(name, val)
    for name, filename, content, content_type in (files or []):
        form.add_field(name, content, filename=filename, content_type=content_type)
    return form


class WebServerTest(AioHTTPTestCase):
    async def get_application(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.wait_lock = asyncio.Lock()
        self.uploaded_entities = []

        def fake_upload(result, discord_user, cache_filename, source='discord'):
            entity = {
                'hash': f'hash{len(self.uploaded_entities) + 1:02d}',
                'source': source,
                'cache_filename': cache_filename,
                'discord_user': discord_user,
            }
            self.uploaded_entities.append(entity)
            return entity

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown, False)

        async def fake_async_scan(filename, locale='auto', for_sale=False):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                executor,
                lambda: scanner.scan_media(str(filename), locale=locale, for_sale=for_sale),
            )

        return web_server.create_app(
            wait_lock=self.wait_lock,
            async_scan_fn=fake_async_scan,
            upload_fn=fake_upload,
            error_fn=discord_bot.improve_error_message,
            turnstile_secret='test-secret',
            cache_dir=self.tmp_dir.name,
        )

    async def test_health_and_cors(self):
        resp = await self.client.get('/api/health', headers={'Origin': 'https://nook.lol'})
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get('Access-Control-Allow-Origin'), 'https://nook.lol')
        self.assertEqual(resp.headers.get('Cache-Control'), 'no-store')
        data = await resp.json()
        self.assertTrue(data['ok'])

        bad_cors = await self.client.options('/api/scan', headers={'Origin': 'https://evil.example'})
        self.assertEqual(bad_cors.status, 204)
        self.assertIsNone(bad_cors.headers.get('Access-Control-Allow-Origin'))

    async def test_turnstile_failure_leaves_no_files(self):
        form = _multipart_upload(
            fields=[('cf-turnstile-response', 'bad-token')],
            files=[('files', 'pic.png', _png_bytes(1280, 720), 'image/png')],
        )
        with mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=False)):
            resp = await self.client.post('/api/scan', data=form, headers=ALLOWED_ORIGIN)
        self.assertEqual(resp.status, 403)
        web_cache = pathlib.Path(self.tmp_dir.name) / 'web'
        self.assertEqual(list(web_cache.glob('**/*')), [])

    async def test_low_res_image_rejected_with_web_tip_and_archived(self):
        form = _multipart_upload(
            fields=[('cf-turnstile-response', 'ok-token')],
            files=[('files', 'small.png', _png_bytes(640, 360), 'image/png')],
        )
        with mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=True)):
            resp = await self.client.post('/api/scan', data=form, headers=ALLOWED_ORIGIN)
        self.assertEqual(resp.status, 400)
        payload = await resp.json()
        self.assertIn('640x360', payload['error'])
        self.assertNotIn('Settings -> Chat', payload['error'])
        failed_dirs = list((pathlib.Path(self.tmp_dir.name) / 'failed').iterdir())
        self.assertEqual(len(failed_dirs), 1)
        self.assertTrue((failed_dirs[0] / 'error.txt').exists())

    async def test_oversize_upload_rejected_413(self):
        form = _multipart_upload(
            fields=[('cf-turnstile-response', 'ok-token')],
            files=[('files', 'huge.mp4', b'0' * 2048, 'video/mp4')],
        )
        with mock.patch.object(web_server, 'MAX_UPLOAD_BYTES', 1024), \
             mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=True)):
            resp = await self.client.post('/api/scan', data=form, headers=ALLOWED_ORIGIN)
        self.assertEqual(resp.status, 413)
        web_cache = pathlib.Path(self.tmp_dir.name) / 'web'
        self.assertEqual(list(web_cache.glob('**/*')), [])

    async def test_per_ip_rate_limit_and_active_job_and_queue_depth(self):
        # Test per-IP active job lock while wait_lock is held.
        async with self.wait_lock:
            form1 = _multipart_upload(
                fields=[('cf-turnstile-response', 'ok-token')],
                files=[('files', 'r.jpg', _png_bytes(1280, 720), 'image/png')],
            )
            with mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=True)):
                r1 = await self.client.post('/api/scan', data=form1, headers=ALLOWED_ORIGIN)
            self.assertEqual(r1.status, 202)

            form2 = _multipart_upload(
                fields=[('cf-turnstile-response', 'ok-token')],
                files=[('files', 'r2.jpg', _png_bytes(1280, 720), 'image/png')],
            )
            with mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=True)):
                r2 = await self.client.post('/api/scan', data=form2, headers=ALLOWED_ORIGIN)
            self.assertEqual(r2.status, 429)
            self.assertIn('already have a scan', (await r2.json())['error'])

            # Queue depth limit from another IP.
            with mock.patch.object(web_server, 'MAX_QUEUE_DEPTH', 1), \
                 mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=True)):
                form3 = _multipart_upload(
                    fields=[('cf-turnstile-response', 'ok-token')],
                    files=[('files', 'r3.jpg', _png_bytes(1280, 720), 'image/png')],
                )
                r3 = await self.client.post('/api/scan', data=form3, headers={'Origin': 'https://nook.lol', 'CF-Connecting-IP': '203.0.113.11'})
                self.assertEqual(r3.status, 503)

        # Wait briefly for queued job to finish scanning (will fail because blank image).
        job_id = (await r1.json())['job_id']
        for _ in range(30):
            st = await (await self.client.get(f'/api/status/{job_id}')).json()
            if st['status'] in ('done', 'error'):
                break
            await asyncio.sleep(0.1)

        # Sliding 5-per-10-min limit on a fresh IP.
        ip3 = {'Origin': 'https://nook.lol', 'CF-Connecting-IP': '203.0.113.12'}
        with mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=True)):
            for idx in range(5):
                f = _multipart_upload(
                    fields=[('cf-turnstile-response', 'ok-token')],
                    files=[('files', f'p{idx}.png', _png_bytes(1280, 720), 'image/png')],
                )
                resp = await self.client.post('/api/scan', data=f, headers=ip3)
                self.assertEqual(resp.status, 202)
                jid = (await resp.json())['job_id']
                for _ in range(40):
                    st = await (await self.client.get(f'/api/status/{jid}')).json()
                    if st['status'] in ('done', 'error'):
                        break
                    await asyncio.sleep(0.05)

            f_over = _multipart_upload(
                fields=[('cf-turnstile-response', 'ok-token')],
                files=[('files', 'p5.png', _png_bytes(1280, 720), 'image/png')],
            )
            r_over = await self.client.post('/api/scan', data=f_over, headers=ip3)
            self.assertEqual(r_over.status, 429)
            self.assertIn('Rate limit', (await r_over.json())['error'])

    async def test_real_reactions_single_and_multi_image_scan(self):
        sample_path = pathlib.Path(__file__).resolve().parent / 'examples' / 'reactions.jpg'
        if not sample_path.exists():
            self.skipTest('examples/reactions.jpg not present')
        raw = sample_path.read_bytes()

        # Single image scan.
        form_single = _multipart_upload(
            fields=[('cf-turnstile-response', 'ok-token')],
            files=[('files', 'reactions.jpg', raw, 'image/jpeg')],
        )
        with mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=True)):
            r = await self.client.post('/api/scan', data=form_single, headers={'Origin': 'https://nook.lol', 'CF-Connecting-IP': '203.0.113.20'})
        self.assertEqual(r.status, 202)
        job_id = (await r.json())['job_id']

        st = {}
        for _ in range(60):
            st = await (await self.client.get(f'/api/status/{job_id}')).json()
            if st['status'] in ('done', 'error'):
                break
            await asyncio.sleep(0.1)
        self.assertEqual(st['status'], 'done', st)
        self.assertEqual(st['count'], 31)
        self.assertEqual(st['mode'], 'reactions')
        self.assertTrue((pathlib.Path(self.tmp_dir.name) / 'web' / job_id).exists())

        # Multi-image scan (2 copies -> frame_0.png, frame_1.png -> frame_%d.png).
        form_multi = _multipart_upload(
            fields=[('cf-turnstile-response', 'ok-token'), ('for_sale', 'true')],
            files=[
                ('files', 'reactions_1.jpg', raw, 'image/jpeg'),
                ('files', 'reactions_2.jpg', raw, 'image/jpeg'),
            ],
        )
        with mock.patch.object(web_server, 'verify_turnstile', new=mock.AsyncMock(return_value=True)):
            r_multi = await self.client.post('/api/scan', data=form_multi, headers={'Origin': 'https://nook.lol', 'CF-Connecting-IP': '203.0.113.21'})
        self.assertEqual(r_multi.status, 202)
        job_id2 = (await r_multi.json())['job_id']

        st2 = {}
        for _ in range(60):
            st2 = await (await self.client.get(f'/api/status/{job_id2}')).json()
            if st2['status'] in ('done', 'error'):
                break
            await asyncio.sleep(0.1)
        self.assertEqual(st2['status'], 'done', st2)
        self.assertEqual(st2['count'], 31)
        self.assertEqual(self.uploaded_entities[-1]['source'], 'web')
        self.assertTrue((pathlib.Path(self.tmp_dir.name) / 'web' / job_id2 / 'frame_0.png').exists())
        self.assertTrue((pathlib.Path(self.tmp_dir.name) / 'web' / job_id2 / 'frame_1.png').exists())


if __name__ == '__main__':
    unittest.main()
