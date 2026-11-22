import asyncio
import concurrent.futures
import datetime
import logging as stdlib_logging
import os
import pathlib
import re
import time
from typing import Optional
import zlib

from absl import app, logging
import discord
from discord import option
from discord.enums import IntegrationType, InteractionContextType
from discord.ext import commands
from google.cloud import datastore
import hashids
import simple_colors

import catalog
import constants
import scanner
import web_server

ERROR_EMOJI = ':exclamation:'
SUCCESS_EMOJI = ':tada:'
SCANNING_EMOJI = ':mag:'
WAIT_EMOJI = ':hourglass:'

CACHE_MAX_AGE_SECONDS = 14 * 24 * 3600  # 14 days
CACHE_MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 500 MiB

WAIT_LOCK = asyncio.Lock()
SCAN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_pending_scans = 0

intents = discord.Intents.default()
bot = commands.Bot(intents=intents)

hashid_client = hashids.Hashids(salt=constants.SALT, min_length=6)
_datastore_client = None
_web_runner = None
_UNINDEXED_FIELDS = (
    'hash', 'data', 'locale', 'type', 'source',
    'discord_user', 'discord_user_name', 'cache_filename', 'unmatched',
)


def _get_datastore_client() -> datastore.Client:
    global _datastore_client
    if _datastore_client is None:
        _datastore_client = datastore.Client.from_service_account_json('catalog-scanner.json')
    return _datastore_client


def _prune_cache_dir(cache_dir: pathlib.Path) -> None:
    """Bounds cache directory size and age while keeping recent uploads for debugging."""
    try:
        now = time.time()
        entries = []
        for entry in cache_dir.iterdir():
            if not entry.is_file():
                continue
            stat = entry.stat()
            if (now - stat.st_mtime) > CACHE_MAX_AGE_SECONDS:
                entry.unlink(missing_ok=True)
            else:
                entries.append((stat.st_mtime, stat.st_size, entry))

        total_bytes = sum(size for _, size, _ in entries)
        if total_bytes > CACHE_MAX_TOTAL_BYTES:
            for _, size, entry in sorted(entries, key=lambda x: x[0]):
                entry.unlink(missing_ok=True)
                total_bytes -= size
                if total_bytes <= CACHE_MAX_TOTAL_BYTES:
                    break
    except OSError:
        logging.warning('Failed to prune cache directory.', exc_info=True)


def upload_to_datastore(
    result,
    discord_user: Optional[discord.User],
    cache_filename: str,
    source: str = 'discord',
) -> datastore.Entity:
    datastore_client = _get_datastore_client()

    temp_key = datastore_client.key('Catalog')
    key = datastore_client.allocate_ids(temp_key, 1)[0]
    entity = datastore.Entity(key, exclude_from_indexes=_UNINDEXED_FIELDS)

    key_hash = hashid_client.encode(key.id)
    raw_data = '\n'.join(result.items).encode('utf-8')
    compressed_data = zlib.compress(raw_data, level=6)
    catalog_data = compressed_data if len(compressed_data) < len(raw_data) else raw_data
    entity.update(
        {
            'hash': key_hash,
            'data': catalog_data,
            'locale': result.locale,
            'type': result.mode.name.lower(),
            'source': source,
            'created': datetime.datetime.now(datetime.timezone.utc),
            'cache_filename': cache_filename,
        }
    )
    if discord_user is not None:
        entity['discord_user'] = discord_user.id
        entity['discord_user_name'] = discord_user.name
    if result.unmatched:
        unmatched_data = '\n'.join(result.unmatched).encode('utf-8')
        entity['unmatched'] = unmatched_data

    datastore_client.put(entity)
    return entity


async def handle_scan(
    ctx: discord.ApplicationContext,
    attachment: discord.Attachment,
    filetype: str,
    locale: str = 'auto',
    for_sale: bool = False,
) -> None:
    """Downloads the file, runs the scans and uploads the results, while updating the user along the way."""
    await reply(ctx, f'{SCANNING_EMOJI} Scan started, your results will be ready soon!')

    tmp_dir = pathlib.Path('cache')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_name = pathlib.Path(attachment.filename or f'upload.{filetype}').name
    logging.info('Downloading attachment %s (%s)', attachment.id, safe_name)
    tmp_file = tmp_dir / f'{attachment.id}_{safe_name}'
    await attachment.save(tmp_file)

    try:
        result = await async_scan(tmp_file, locale=locale, for_sale=for_sale)
    except AssertionError as e:
        logging.warning('Scan rejected for %s (%s): %s', ctx.user, tmp_file.name, e)
        error_message = improve_error_message(str(e))
        await reply(ctx, f'{ERROR_EMOJI} Failed to scan: {error_message}')
        return
    except Exception:
        logging.exception('Unexpected scan error on %s.', tmp_file.name)
        await reply(
            ctx,
            f'{ERROR_EMOJI} Failed to scan media. Make sure you have a valid {filetype}.',
        )
        return
    finally:
        _prune_cache_dir(tmp_dir)

    if not result.items:
        logging.warning('Scan found 0 items for %s (%s)', ctx.user, tmp_file.name)
        await reply(ctx, f'{ERROR_EMOJI} Did not find any items.')
        return

    try:
        entity = upload_to_datastore(result, ctx.user, tmp_file.name)
    except Exception:
        logging.exception('Failed to upload scan results to Datastore for %s (%s).', ctx.user, tmp_file.name)
        await reply(
            ctx,
            f'{ERROR_EMOJI} Found {len(result.items)} items, but failed to save results. Please try again.',
        )
        return

    url = 'https://nook.lol/{}'.format(entity['hash'])
    logging.info('Found %s items with %s [%s]: %s', len(result.items), result.mode, result.locale, url)
    await reply(
        ctx,
        f'{SUCCESS_EMOJI} Found {len(result.items)} items in your {filetype}.\nResults: {url}',
    )


async def async_scan(
    filename: os.PathLike,
    locale: str = 'auto',
    for_sale: bool = False,
) -> scanner.ScanResult:
    """Runs the scan asynchronously on the shared scan executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        SCAN_EXECUTOR,
        lambda: scanner.scan_media(str(filename), locale=locale, for_sale=for_sale),
    )


async def reply(ctx: discord.ApplicationContext, message: str) -> None:
    """Sends an ephemeral interaction response or followup, falling back to DM on expired tokens."""
    try:
        await ctx.respond(content=message, ephemeral=True)
    except discord.HTTPException as err:
        logging.warning('Interaction reply failed for %s (%s); trying DM fallback. Message: %s', ctx.user, err, message)
        try:
            await ctx.user.send(message)
        except discord.HTTPException:
            logging.warning('DM fallback also failed for %s', ctx.user)


def improve_error_message(message: str, source: str = 'discord') -> str:
    """Adds some more details to the error message."""
    if 'is too long' in message:
        message += ' Make sure you scroll with the correct analog stick (see instructions),'
        message += ' and trim the video around the start and end of the scrolling.'
    if 'scrolling too slowly' in message:
        message += ' Make sure you hold down the *right* analog stick.'
        message += ' See https://x.com/CatalogScanner/status/1261737975244865539'
    if 'scrolling inconsistently' in message:
        message += ' Please scroll once, from start to finish, in one direction only.'
    if 'Invalid video' in message:
        message += ' Make sure the video is exported directly from your Nintendo Switch '
        message += 'and that you\'re scrolling through a supported page. See nook.lol'
    if 'not showing catalog or recipes' in message:
        message += ' Make sure to record the video with your Switch using the capture button.'
    if 'x224' in message:
        message += '\n(It seems like you\'re downloading the video from your Facebook and '
        message += 're-posting it; try downloading it directly from your Switch instead)'
    if source == 'web' and ('x360' in message or 'x480' in message):
        message += '\nIt looks like your video was compressed before uploading; '
        message += 'export it directly from your Switch Album (Send to Smart Device or USB-C/microSD) '
        message += 'rather than saving it from Discord or a chat app.'
    elif 'x360' in message:
        message += '\nIt looks like you might have Data Saving Mode enabled; '
        message += 'go to *Settings -> Chat*, uncheck Data Saving Mode and make sure *Video Uploads* is set to **Best Quality**.'
    elif 'x480' in message:
        message += '\nIt seems like Discord might have compressed your video; '
        message += 'go to *Settings -> Chat* and set *Video Uploads* to **Best Quality**.'
    elif 'Invalid resolution' in message:
        message += '\n(Make sure you are recording and sending directly from the Switch)'
    if 'Pictures Mode' in message:
        message += ' Press X to switch to list mode and try again!'
    if 'blocking a reaction' in message:
        message += ' Make sure to move the cursor to an empty slot or the top right corner, '
        message += 'otherwise your results may not be accurate.'
    if 'Workbench scanning' in message:
        message += ' Please use the DIY Recipes phone app instead (beige background).'
    if 'catalog is not supported' in message:
        message += ' Please use the Catalog phone app instead (yellow background).'
    if 'Incomplete critter scan' in message:
        message += ' Make sure to fully capture the leftmost and rightmost sides of the page.'
    if 'not uploaded directly' in message:
        message += ' Make sure to record and download the video using the Switch\'s media gallery.'
    return message


@bot.slash_command(
    name='ping',
    description='Responds with the bot latency.',
)
async def ping(ctx: discord.ApplicationContext):
    latency = bot.latency * 1000  # Convert to milliseconds
    await ctx.respond(f'Pong! Latency is {latency:.2f} ms')


@bot.slash_command(
    name='scan',
    description='Extracts your Animal Crossing items (catalog, recipes, critters, reactions, music).',
    integration_types=[IntegrationType.user_install, IntegrationType.guild_install],
    contexts=[InteractionContextType.guild, InteractionContextType.bot_dm, InteractionContextType.private_channel],
)
@option(
    'attachment',
    discord.Attachment,
    description='The video or image to scan',
    required=True,
)
@option(
    'locale',
    str,
    description='Game language (auto-detected by default)',
    choices=list(catalog.LOCALE_MAP),
    default='auto',
    required=False,
)
@option(
    'for_sale',
    bool,
    description='Only include purchasable catalog items',
    default=False,
    required=False,
)
async def scan(
    ctx: discord.ApplicationContext,
    attachment: discord.Attachment,
    locale: str = 'auto',
    for_sale: bool = False,
):
    logging.info(
        'Got request from %s with type %r (locale=%s, for_sale=%s)',
        ctx.user, attachment.content_type, locale, for_sale,
    )

    if not attachment or not attachment.content_type:
        await reply(ctx, f'{ERROR_EMOJI} Invalid or no attachment.')
        return

    filetype, _, _ = attachment.content_type.partition('/')  # {type}/{format}
    if filetype not in ('video', 'image'):
        logging.info('Invalid attachment type %r, skipping.', attachment.content_type)
        await reply(ctx, f'{ERROR_EMOJI} The attachment needs to be a valid video or image file.')
        return

    await ctx.interaction.response.defer(ephemeral=True, invisible=False)

    # Queue system that handles requests one at a time.
    global _pending_scans
    ahead = _pending_scans + (1 if (_pending_scans == 0 and WAIT_LOCK.locked()) else 0)
    _pending_scans += 1
    try:
        if ahead > 0:
            logging.info('%s (%s) is in queue position %s', ctx.user, attachment.id, ahead)
            await reply(ctx, f'{WAIT_EMOJI} You are #{ahead} in the queue, your scan will start soon.')

        async with WAIT_LOCK:
            await handle_scan(ctx, attachment, filetype, locale=locale, for_sale=for_sale)
    finally:
        _pending_scans -= 1


@bot.event
async def on_ready():
    global _web_runner
    assert bot.user, 'Failed to login'
    logging.info('Bot logged in as %s', bot.user)
    if _web_runner is None:
        _web_runner = await web_server.start_server(
            wait_lock=WAIT_LOCK,
            async_scan_fn=async_scan,
            upload_fn=upload_to_datastore,
            error_fn=improve_error_message,
            turnstile_secret=getattr(constants, 'TURNSTILE_SECRET', ''),
        )


_LOG_PREFIX_RE = re.compile(r'^(\w\d{4} [\d:]{8})\.\d+\s+\d+\s+')


class CompactFormatter(logging.PythonFormatter):
    def format(self, record):
        return _LOG_PREFIX_RE.sub(r'\1 ', super().format(record), count=1)


class ColorFormatter(CompactFormatter):
    COLORS = {
        'DEBUG': (simple_colors.cyan, 'dim'),
        'INFO': (simple_colors.black, 'bright'),
        'WARNING': (simple_colors.yellow, 'dim'),
        'ERROR': (simple_colors.red, 'bright'),
        'CRITICAL': (simple_colors.magenta, 'blink'),
    }

    def format(self, record):
        log = super().format(record)
        prefix, sep, message = log.partition(']')
        entry = self.COLORS.get(record.levelname)
        if not entry:
            return log
        color, style = entry
        return color(prefix + sep, style) + message


def main(argv):
    del argv  # unused
    bot.run(constants.DISCORD_TOKEN)


if __name__ == '__main__':
    # Write plain logs to disk and colored logs to console/journal.
    file_handler = stdlib_logging.FileHandler('logs.txt')
    file_handler.setFormatter(CompactFormatter())
    logging.get_absl_logger().addHandler(file_handler)  # type: ignore
    logging.get_absl_handler().setFormatter(ColorFormatter())

    # Disable noise discord logs.
    stdlib_logging.getLogger('discord.client').setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger('discord.gateway').setLevel(stdlib_logging.WARNING)

    app.run(main)
