import os
import platform
import contextlib

import discord
from discord.ext import commands
from discord import option
from dotenv import load_dotenv

import scanner
load_dotenv()
DISCORD_TOKEN  = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
bot = commands.Bot(intents=intents)

async def handle_message(ctx: discord.ApplicationContext, attachment: discord.Attachment, integration: str):
	attch_type = attachment.content_type.split('/') # Attachment.content_type returns a {type}/{file_format} string
	if attch_type[0] != 'video':
		await ctx.respond('The attachment needs to be a valid video file', ephemeral=True)
	else:
		try:
			await ctx.respond('Scan started, your results will be ready soon', ephemeral=True)
			file = await attachment.to_file()
			path = 'dump'
			if not os.path.exists(path):
				os.makedirs(path)
			temp_file = os.path.join(path, file.filename)
			await attachment.save(temp_file)

			result = scanner.scan_media(temp_file)
			with contextlib.suppress(FileNotFoundError):
				os.remove(temp_file)

			# Do cool redirect stuff here!
			await ctx.respond(f"Found {len(result.items)} items in your video.\nResults:", ephemeral=True)
		except Exception as e:
			print(f'Failed to scan video {e}')
			await ctx.respond('Failed to scan video. Make sure you have a valid video file.', ephemeral=True)

@bot.event
async def on_ready():
    print(f"Logged in as a bot {bot.user.name}")
    print(f"discord.py API version: {discord.__version__}")
    print(f"Python version: {platform.python_version()}")
    print(f"Running on: {platform.system()} {platform.release()} ({os.name})")
    print("-------------------")
    
@bot.slash_command(name="scan", description="Extracts your Animal Crossing catalog items and DIY recipes from a recorded video.")
@option("attachment",discord.Attachment,description="The video to scan", required=True)
@option("integration", description="Visit https://nook.lol/ to see available hashtags", required=False)
async def scan(ctx: discord.ApplicationContext, attachment: discord.Attachment, integration: str):
	if attachment:
		await handle_message(ctx, attachment, integration)
	else:
		await ctx.respond('No attachment found.', ephemeral=True)

# EXECUTES THE BOT WITH THE SPECIFIED TOKEN. TOKEN HAS BEEN REMOVED AND USED JUST AS AN EXAMPLE.
bot.run(DISCORD_TOKEN)