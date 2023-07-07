import os
import platform
import contextlib
import hashids
import datetime
from google.cloud import datastore

import discord
from discord.ext import commands
from discord import option

import constants
import scanner

intents = discord.Intents.default()
bot = commands.Bot(intents=intents)

hashid_client = hashids.Hashids(salt=constants.SALT, min_length=6)


def upload_to_datastore(result, discord_user_id=None):
    print(f'Uploading to datastore...')

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
        'discord_user_id': discord_user_id
    })
    if result.unmatched:
        unmatched_data = '\n'.join(result.unmatched).encode('utf-8')
        catalog['unmatched'] = unmatched_data

    datastore_client.put(catalog)
    return catalog


async def handle_message(ctx: discord.ApplicationContext, attachment: discord.Attachment):
    # Attachment.content_type returns a {type}/{file_format} string
    attch_type = attachment.content_type.split('/')[0]
    if attch_type != 'video' and attch_type != 'image':
        await ctx.respond(f'${constants.ERROR_EMOJI} The attachment needs to be a valid video or image file', ephemeral=True)
        return
    
    await ctx.respond(f'${constants.SCANNING_EMOJI} Scan started, your results will be ready soon!', ephemeral=True)
    file = await attachment.to_file()
    path = 'dump'
    if not os.path.exists(path):
        os.makedirs(path)
    temp_file = os.path.join(path, file.filename)
    await attachment.save(temp_file)

    try:
        result = scanner.scan_media(temp_file)
    except Exception as e:
        print(f'Failed to scan media {e}')
        ctx.interaction.edit_original_response(content=f'${constants.ERROR_EMOJI} Failed to scan media. Make sure you have a valid ${attch_type} file')

    with contextlib.suppress(FileNotFoundError):
        os.remove(temp_file)

    catalog = upload_to_datastore(result, ctx.user.id)
    url = 'https://nook.lol/{}'.format(catalog['hash'])
    await ctx.interaction.edit_original_response(content=f'${constants.SUCCESS_EMOJI} Found {len(result.items)} items in your ${attch_type}\nResults: {url}')


@bot.event
async def on_ready():
    print(f'Logged in as a bot {bot.user.name}')
    print(f'discord.py API version: {discord.__version__}')
    print(f'Python version: {platform.python_version()}')
    print(f'Running on: {platform.system()} {platform.release()} ({os.name})')
    print('-------------------')


@bot.slash_command(name='scan', description='Extracts your Animal Crossing catalog items and DIY recipes from a recorded video or image')
@option('attachment', discord.Attachment, description='The media file to scan', required=True)
async def scan(ctx: discord.ApplicationContext, attachment: discord.Attachment, integration: str):
    if attachment:
        await handle_message(ctx, attachment, integration)
    else:
        await ctx.respond(f'${constants.ERROR_EMOJI} No attachment found', ephemeral=True)

bot.run(constants.DISCORD_TOKEN)
