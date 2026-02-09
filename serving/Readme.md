## Serving
### Serve locally

```bash
poetry env activate
poetry install

make serve
```

### Fireplace Detection & Tracking
- The fireplace region is represented by a **4-point polygon**
- The polygon is manually editable through a web UI if the camera is moved

### Image Classification
- Images are cropped using the polygon and warped to a fixed size
- Features are extracted and fed into a **Random Forest classifier**

## API Endpoints
### GET /trigger_predict
- Captures a frame from the video stream
- Runs the full prediction pipeline
- Returns JSON:
```
{
  "label": "high_flame",
  "confidence": 0.93,
  "timestamp": "2026-01-01T20:15:00.000000"
}
```

### POST /predict
- Use last available snapshot
- Runs prediction without touching the stream

### POST /update_polygon
- Updates the active fireplace polygon
- Called by the web UI after user clicks 4 points

## Web Interface
### /home
A lightweight HTML page (home.html) is included to support manual intervention:
Features:
- Displays the latest captured frame
- button to trigger prediction (call /trigger_predict)
- Latest warped ROI (region of interest) image (/tmp/warped.jpg)
- show prediction result

### /update_polygon
Features:
- Displays the latest captured frame
- Allows the user to click 4 corners to redefine the fireplace polygon
- Saves the polygon via /update_polygon
- Button to trigger prediction

### /scheduler
- used to organise data collection

