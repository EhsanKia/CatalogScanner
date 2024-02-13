import asyncio
import concurrent.futures
import logging as stdlib_logging
import os
import datetime
import pathlib

from absl import app, logging
import discord
from discord.ext import commands
from discord import option
from google.cloud import datastore
import hashids

import constants
import scanner
import twitter_downloader.src.twitter_downloader as tvdl

ERROR_EMOJI = ':exclamation:'
SUCCESS_EMOJI = ':tada:'
SCANNING_EMOJI = ':mag:'
WAIT_EMOJI = ':hourglass:'

WAIT_LOCK = asyncio.Lock()

intents = discord.Intents.default()
bot = commands.Bot(intents=intents)

hashid_client = hashids.Hashids(salt=constants.SALT, min_length=6)


def upload_to_datastore(result, discord_user_id=None) -> datastore.Entity:
    datastore_client = datastore.Client.from_service_account_json('catalog-scanner.json')

    temp_key = datastore_client.key('Catalog')
    key = datastore_client.allocate_ids(temp_key, 1)[0]
    catalog = datastore.Entity(key, exclude_from_indexes=['data', 'unmatched'])

    key_hash = hashid_client.encode(key.id)
    catalog_data = '\n'.join(result.items).encode('utf-8')
    catalog.update(
        {
            'hash': key_hash,
            'data': catalog_data,
            'locale': result.locale,
            'type': result.mode.name.lower(),
            'created': datetime.datetime.utcnow(),
            'discord_user': discord_user_id,
        }
    )
    if result.unmatched:
        unmatched_data = '\n'.join(result.unmatched).encode('utf-8')
        catalog['unmatched'] = unmatched_data

    datastore_client.put(catalog)
    return catalog


async def handle_scan(
    ctx: discord.ApplicationContext,
    attachment: discord.Attachment,
    filetype: str,
    url: str,
) -> None:
    """Downloads the file, runs the scans and uploads the results, while updating the user along the way."""
    await reply(ctx, f'{SCANNING_EMOJI} Scan started, your results will be ready soon!')

    tmp_dir = pathlib.Path('cache')

    if url:
        logging.info('Downloading video from %s', url)
        try:
            tmp_file = tmp_dir / f'{ctx.user.id}_video.mp4'
            tvdl.download_twitter_video(url, tmp_file)
        except (Exception, SystemExit):
            logging.exception('Unexpected scan error.')
            await reply(ctx, f'{ERROR_EMOJI} Failed to scan media. Make sure you have a valid {filetype}.')
            return
    else:
        logging.info('Downloading attachment %s', attachment.id)
        file = await attachment.to_file()
        tmp_file = tmp_dir / f'{attachment.id}_{file.filename}'
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        await attachment.save(tmp_file)

    try:
        result = await async_scan(tmp_file)
    except AssertionError as e:
        error_message = improve_error_message(str(e))
        await reply(ctx, f'{ERROR_EMOJI} Failed to scan: {error_message}')
        return
    except Exception:
        logging.exception('Unexpected scan error.')
        await reply(
            ctx,
            f'{ERROR_EMOJI} Failed to scan media. Make sure you have a valid {filetype}.',
        )
        return

    if not result.items:
        await reply(ctx, f'{ERROR_EMOJI} Did not find any items.')
        return

    catalog = upload_to_datastore(result, ctx.user.id)
    url = 'https://nook.lol/{}'.format(catalog['hash'])
    logging.info('Found %s items with %s: %s', len(result.items), result.mode, url)
    await reply(
        ctx,
        f'{SUCCESS_EMOJI} Found {len(result.items)} items in your {filetype}.\nResults: {url}',
    )


async def async_scan(filename: os.PathLike) -> scanner.ScanResult:
    """Runs the scan asynchronously in a thread pool."""
    pool = concurrent.futures.ThreadPoolExecutor()
    future = pool.submit(scanner.scan_media, str(filename))
    return await asyncio.wrap_future(future)


async def reply(ctx: discord.ApplicationContext, message: str) -> None:
    """Responds with an ephemeral message, or updates the existing message."""
    if not ctx.interaction.response.is_done():
        await ctx.interaction.response.send_message(content=message, ephemeral=True)
    else:
        await ctx.interaction.edit_original_response(content=message)


def improve_error_message(message: str) -> str:
    """Adds some more details to the error message."""
    if 'is too long' in message:
        message += ' Make sure you scroll with the correct analog stick (see instructions),'
        message += ' and trim the video around the start and end of the scrolling.'
    if 'scrolling too slowly' in message:
        message += ' Make sure you hold down the *right* analog stick.'
        message += ' See https://twitter.com/CatalogScanner/status/1261737975244865539'
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
    if 'x360' in message:
        message += '\nIt looks like you might have Data Saving Mode enabled; '
        message += 'go to *Settings -> Text & Media*, uncheck Data Saving Mode and make sure *Video Uploads* is set to **Best Quality**.'
    elif 'x480' in message:
        message += '\nIt seems like Discord might have compressed your video; '
        message += 'go to *Settings -> Text & Media* and set *Video Uploads* to **Best Quality**.'
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
    name='scan',
    description='Extracts your Animal Crossing items (catalog, recipes, critters, reactions, music).',
)
@option('url', str, description='The url of a video to scan', required=False)
@option(
    'attachment',
    discord.Attachment,
    description='The video or image to scan',
    required=False,
)
async def scan(ctx: discord.ApplicationContext, url: str, attachment: discord.Attachment):
    if attachment and url:
        await reply(ctx, f'{ERROR_EMOJI} Please provide either an attachment or a URL, not both.')
        return

    if attachment:
        assert attachment.content_type
        filetype, _, _ = attachment.content_type.partition('/')  # {type}/{format}
        if filetype not in ('video', 'image'):
            logging.info('Invalid attachment type %r, skipping.', attachment.content_type)
            await reply(ctx, f'{ERROR_EMOJI} The attachment needs to be a valid video or image file.')
            return
    elif url:
        filetype = 'url'
    else:
        await reply(ctx, f'{ERROR_EMOJI} No attachment or url found.')
        return

    logging.info('Got request from %s with type %r', ctx.user, filetype)

    # Have a queue system that handles requests one at a time.
    if WAIT_LOCK.locked and (waiters := WAIT_LOCK._waiters):  # type: ignore
        logging.info('%s (%s) is in queue position %s', ctx.user, attachment.id, len(waiters))
        await reply(ctx, f'{WAIT_EMOJI} You are #{len(waiters)} in the queue, your scan will start soon.')
    async with WAIT_LOCK:
        await handle_scan(ctx, attachment, filetype, url)


@bot.event
async def on_ready():
    assert bot.user, 'Failed to login'
    logging.info('Bot logged in as %s', bot.user)


def main(argv):
    del argv  # unused
    bot.run(constants.DISCORD_TOKEN)


if __name__ == '__main__':
    # Write logs to file.
    file_handler = stdlib_logging.FileHandler('logs.txt')
    logging.get_absl_logger().addHandler(file_handler)  # type: ignore
    # Disable noise discord logs.
    stdlib_logging.getLogger('discord.client').setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger('discord.gateway').setLevel(stdlib_logging.WARNING)

    app.run(main)
