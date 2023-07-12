import asyncio
import concurrent.futures
import logging as stdlib_logging
import os
import contextlib
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

ERROR_EMOJI = "\U00002757"
SUCCESS_EMOJI = "\U0001F389"
SCANNING_EMOJI = "\U0001F50D"

intents = discord.Intents.default()
bot = commands.Bot(intents=intents)

hashid_client = hashids.Hashids(salt=constants.SALT, min_length=6)


def upload_to_datastore(result, discord_user_id=None) -> datastore.Entity:
    datastore_client = datastore.Client.from_service_account_json(
        'catalog-scanner.json')

    temp_key = datastore_client.key('Catalog')
    key = datastore_client.allocate_ids(temp_key, 1)[0]
    catalog = datastore.Entity(key, exclude_from_indexes=['data', 'unmatched'])

    key_hash = hashid_client.encode(key.id)
    catalog_data = '\n'.join(result.items).encode('utf-8')
    catalog.update({
        'hash': key_hash,
        'data': catalog_data,
        'locale': result.locale,
        'type': result.mode.name.lower(),
        'created': datetime.datetime.utcnow(),
        'discord_user': discord_user_id,
    })
    if result.unmatched:
        unmatched_data = '\n'.join(result.unmatched).encode('utf-8')
        catalog['unmatched'] = unmatched_data

    datastore_client.put(catalog)
    return catalog


async def handle_message(ctx: discord.ApplicationContext, attachment: discord.Attachment) -> None:
    if not attachment:
        await ctx.respond(f'{ERROR_EMOJI} No attachment found.', ephemeral=True)
        return

    # Attachment.content_type returns a {type}/{file_format} string
    assert attachment.content_type
    filetype, _, _ = attachment.content_type.partition('/')
    if filetype not in ('video', 'image'):
        await ctx.respond(f'{ERROR_EMOJI} The attachment needs to be a valid video or image file', ephemeral=True)
        return

    await ctx.respond(f'{SCANNING_EMOJI} Scan started, your results will be ready soon!', ephemeral=True)
    file = await attachment.to_file()
    tmp_dir = pathlib.Path('cache')
    tmp_file = tmp_dir / f'{attachment.id}_{file.filename}'
    tmp_file.parent.mkdir(parents=True, exist_ok=True)
    await attachment.save(tmp_file)

    try:
        result = await async_scan(tmp_file)
    except AssertionError as e:
        error_message = improve_error_message(str(e))
        await ctx.edit(content=f'{ERROR_EMOJI} Failed to scan: {error_message}')
        return
    except Exception:
        logging.exception('Unexpected scan error.')
        ctx.edit(
            content=f'{ERROR_EMOJI} Failed to scan media. Make sure you have a valid ${filetype} file.')
        return

    if not result.items:
        await ctx.edit(content=f'${ERROR_EMOJI} Did not find any items.')
        return

    with contextlib.suppress(FileNotFoundError):
        os.remove(tmp_file)

    catalog = upload_to_datastore(result, ctx.user.id)
    url = 'https://nook.lol/{}'.format(catalog['hash'])
    logging.info('Found %s items with %s: %s',
                 len(result.items), result.mode, url)
    await ctx.edit(content=f"{SUCCESS_EMOJI} Found {len(result.items)} items in your ${filetype}.\nResults: {url}")


async def async_scan(filename: os.PathLike) -> scanner.ScanResult:
    """Runs the scan asynchronously in a thread pool."""
    pool = concurrent.futures.ThreadPoolExecutor()
    future = pool.submit(scanner.scan_media, str(filename))
    return await asyncio.wrap_future(future)


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
    if '640x360' in message:
        message += '\nIt seems like Discord might have compressed your video; go to *Settings -> Text & Media* and set *Video Uploads* to **Best Quality**.'
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
@option('attachment', discord.Attachment, description='The video to scan', required=True)
async def scan(ctx: discord.ApplicationContext, attachment: discord.Attachment):
    logging.info('Got request from %s', ctx.user)
    await handle_message(ctx, attachment)


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
    logging.get_absl_logger().addHandler(file_handler)
    # Disable noise discord logs.
    stdlib_logging.getLogger('discord.client').setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger('discord.gateway').setLevel(
        stdlib_logging.WARNING)

    app.run(main)
