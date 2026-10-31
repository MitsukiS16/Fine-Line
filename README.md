# Fine Line

Turn a photo into a detailed line drawing, right in your browser.

Fine Line uses a small AI model to find the lines that matter and leave out the background, then redraws them as smooth pen strokes that taper at the ends. It runs entirely on your device, so there is no account and no upload.

## Features

- Choose, drag in or paste an image
- **Detail**: from only the main lines to many fine ones
- **Clean up**: removes texture and tiny marks
- **Line thickness**: from hair-thin to bold
- **Grid**: an optional grid with an adjustable number of squares, useful for copying a drawing by hand
- **Paper**: white or cream
- **Download** as PNG or as SVG, which scales to any size. Files are named after their settings, so different versions are easy to tell apart.

## How to use it

1. Download this repository.
2. Open `index.html` in your browser.
3. Choose an image and wait for the drawing. Then adjust the sliders and download the versions you like.

No installation or server is needed. `index.html`, `fineline.js` and `ai-bundle.js` must stay in the same folder.

## Good to know

- Each new image takes about 30 seconds on a laptop, and longer on slower devices. Moving the sliders afterward is almost instant.
- The first time you use it, the browser loads the 37 MB AI bundle, which may be slow on a mobile connection.
- If the AI step can't run, the app falls back to a basic tracer and shows a message. Those drawings may include the background.
- Photos with one clear subject work best. Busy patterns can come out as many small strokes.
- So far it has only been tested in Firefox on a desktop computer.

## How it works

1. The AI model produces a rough line drawing of the image.
2. A separate tracer finds precise lines in the photo and keeps only those near the AI's lines, which removes most of the background.
3. Those lines are thinned, joined into long strokes, smoothed, and drawn with varying width.

| File | Purpose |
| --- | --- |
| `index.html` | The page, its controls and the downloads |
| `fineline.js` | The line tracer |
| `ai-bundle.js` | Generated file containing the AI engine and model |

## Credits

- Line-art model: [Informative Drawings](https://github.com/carolineec/informative-drawings) by Caroline Chan et al. (MIT License), using the [ONNX conversion by rocca](https://huggingface.co/rocca/informative-drawings-line-art-onnx)
- AI engine: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.17.3 by Microsoft (MIT License)

