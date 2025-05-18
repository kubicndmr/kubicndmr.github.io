#!/bin/bash

FAVICON_FOLDER='assets/img/favicons'

SOURCE_IMAGE="$FAVICON_FOLDER/web-app-manifest-512x512.jpg"
rm "$FAVICON_FOLDER/browserconfig.xml" "$FAVICON_FOLDER/site.webmanifest"
ffmpeg -i "$SOURCE_IMAGE" -vf "scale=16:16" "$FAVICON_FOLDER/favicon-16x16.png"
ffmpeg -i "$SOURCE_IMAGE" -vf "scale=32:32" "$FAVICON_FOLDER/favicon-32x32.png"
ffmpeg -i "$SOURCE_IMAGE" -vf "scale=192x192" "$FAVICON_FOLDER/android-chrome-192x192.png"
cp "$SOURCE_IMAGE" "$FAVICON_FOLDER/android-chrome-512x512"
ffmpeg -i "$SOURCE_IMAGE" -vf "scale=150:150" "$FAVICON_FOLDER/mstile-150x150.png"