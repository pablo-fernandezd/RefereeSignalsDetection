import yt_dlp
import os

url = "https://www.youtube.com/watch?v=tRrhcSkViGw" # Example URL
# Corrected path relative to src/utils/
output_dir = "../../data/input_videos"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Configuración de yt-dlp con cookies de Firefox
ydl_opts = {
    # Path template for output files
    "outtmpl": os.path.join(output_dir, '%(title)s.%(ext)s'),
    # Final video format
    "merge_output_format": "mp4",
    # Use Firefox cookies
    "cookiesfrombrowser": ('firefox',),
    "quiet": False, # Show standard output
    "noplaylist": True, # Only download single video
    # Preferred format selection
    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
}

print(f"Starting download for: {url}")
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download complete.")
except Exception as e:
    print(f"An error occurred during download: {e}")